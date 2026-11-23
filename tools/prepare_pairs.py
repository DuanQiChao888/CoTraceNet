"""Build source-grouped CoTraceNet detection benchmarks from a local BOSSbase ZIP."""
import argparse
import csv
import hashlib
import io
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
import random
import re
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, convolve1d, uniform_filter
from scipy.special import xlogy


CONDITIONS = ("HILL_04", "HILL_01", "WOW_04", "WOW_01", "JPEG_03")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def archive_members(archive):
    members = {}
    for info in sorted(archive.infolist(), key=lambda item: item.header_offset):
        name = Path(info.filename).name
        if re.fullmatch(r"[1-9][0-9]*\.pgm", name):
            if name in members:
                raise ValueError(f"Duplicate image name in archive: {name}")
            members[name] = info.filename
    return members


def historical_cohorts(members):
    """Rebuild the historical source IDs from archive order and public seeds."""
    if set(members) != {f"{n}.pgm" for n in range(1, 10001)}:
        raise ValueError("The study split requires the original 10,000-image BOSSbase 1.01 ZIP")
    names = list(members)
    rng = random.Random(20261122)
    pool = []
    for i in range(64):
        start = rng.randint(i*len(names)//64, (i+1)*len(names)//64-8)
        pool.extend(names[start:start+8])
    numeric = lambda name: int(Path(name).stem)
    pool.sort(key=numeric)
    perm = np.random.default_rng(17).permutation(512)
    groups = np.split(perm, [307, 409])
    selections = []
    for index, (split, size) in enumerate((("train", 160), ("val", 48))):
        candidates = sorted(pool[i] for i in groups[index])
        rng = np.random.default_rng(np.random.SeedSequence([20261122, index]))
        selections.extend((split, name) for name in sorted(rng.choice(candidates, size, replace=False)))
    used = set(pool)
    for seed, count in ((20261123, 128), (20261124, 256), (20261125, 256), (20261126, 256)):
        candidates = sorted(set(names)-used, key=numeric)
        chosen = np.random.default_rng(seed).choice(candidates, count, replace=False)
        used.update(chosen)
    selections.extend(("test", name) for name in sorted(chosen, key=numeric))
    # One aggregate receipt detects repacked archives with a different member order.
    recipe = "".join(f"{split}\t{name}\n" for split,name in selections).encode("ascii")
    if sha(recipe) != "2d6677edfabf225d9a6b734311b357dca5ce38358170ff9d738421b33ded9d41":
        raise ValueError("The archive member order differs from the historical study archive")
    eligible = sorted(set(names) - used, key=numeric)
    assert len(eligible) == 8592
    fresh = np.random.default_rng(20261127).choice(eligible, 256, replace=False)
    test = [("test", name) for name in sorted(fresh, key=numeric)]
    recipe = "".join(f"{split}\t{name}\n" for split, name in test).encode("ascii")
    if sha(recipe) != "d7593b975fa1192699b7b01d6b05c5bcd33434fef0c0b0222bb7d62c190b0c74":
        raise ValueError("The fresh test split differs from the locked protocol")
    return selections, test, used.union(fresh)


def study_split(members, cohort="original"):
    original, fresh, _ = historical_cohorts(members)
    if cohort == "original":
        return original
    if cohort == "fresh":
        return fresh
    raise ValueError(f"Unknown test cohort: {cohort}")


def private_split(path, members):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        items = list(csv.DictReader(stream))
    selected = [(r["split"], r["source"]) for r in items]
    if (not selected or any(s not in {"train", "val", "test"} or n not in members for s,n in selected)
            or len({n for _,n in selected}) != len(selected)):
        raise ValueError("Private split must contain unique archive filenames and split train, val or test")
    return selected


def hill_cost(image):
    kernel = [[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]
    residual = convolve(image.astype(np.float64), kernel, mode="reflect")
    activity = np.maximum(uniform_filter(np.abs(residual), 3, mode="reflect"), 1e-8)
    return uniform_filter(1./activity, 15, mode="reflect").clip(1e-12, 1e10)


def separable(image, vertical, horizontal):
    filtered = convolve1d(image, vertical, axis=0, mode="constant")
    return convolve1d(filtered, horizontal, axis=1, mode="constant")


def wow_cost(image):
    # Daubechies-8 high-pass taps at the precision of the DDE Lab WOW reference.
    high = np.array([
        -0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837,
        0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266,
        0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940,
        0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768,
    ])
    low = high[::-1] * ((-1.) ** np.arange(len(high)))
    padded = np.pad(image.astype(np.float64), 16, mode="symmetric")
    cost = np.zeros(image.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for v,h in ((low,high), (high,low), (high,high)):
            residual = separable(padded, v, h)
            suitability = separable(np.abs(residual), np.abs(v[::-1]), np.abs(h[::-1]))
            cost += 1./np.roll(suitability, (1,1), axis=(0,1))[16:-16,16:-16]
    return np.minimum(np.nan_to_num(cost,nan=1e10,posinf=1e10),1e10)


def spatial_draw(image, cost, random_field, target):
    def probabilities(lam):
        weight = np.exp(-lam * cost)
        up = weight*(image<255)
        down = weight*(image>0)
        den = 1.+up+down
        u,d,stay = up/den,down/den,1./den
        entropy = -np.mean(xlogy(u,u)+xlogy(d,d)+xlogy(stay,stay))/np.log(2.)
        return u,d,float(entropy)
    lo,hi = 0.,1.
    while probabilities(hi)[2] > target:
        hi *= 2
    for _ in range(70):
        mid = (lo+hi)/2
        up,down,h = probabilities(mid)
        if abs(h-target) <= 1e-5:
            delta = (random_field<up).astype(np.int16)-(random_field>=1.-down).astype(np.int16)
            return (image.astype(np.int16)+delta).astype(np.uint8)
        if h>target:
            lo = mid
        else:
            hi = mid
    raise ValueError("Target entropy did not converge")


def jpeg_pair(image, source_hash):
    import conseal
    import jpeglib

    jpeglib.version.set("6b")
    seed = int(sha(("q85-juniward-v1/"+source_hash).encode())[:8],16)
    with tempfile.TemporaryDirectory(prefix="ct-jpeg-") as directory:
        cj, sj = (str(Path(directory)/name) for name in ("cover.jpg","stego.jpg"))
        jpeglib.from_spatial(image[...,None]).write_spatial(cj, qt=85)
        cover = jpeglib.read_spatial(cj).spatial[:,:,0].copy()
        jpeg = jpeglib.read_dct(cj)
        coeff = conseal.juniward.simulate_single_channel(
            x0=cover,y0=jpeg.Y.copy(),qt=jpeg.qt[0].copy(),alpha=.3,seed=seed,
            implementation=conseal.juniward.JUNIWARD_ORIGINAL)
        jpeg.Y = coeff
        jpeg.write_dct(sj)
        if not np.array_equal(jpeglib.read_dct(sj).Y,coeff):
            raise ValueError("JPEG coefficient round-trip mismatch")
        stego = jpeglib.read_spatial(sj).spatial[:,:,0].copy()
    return cover,stego


@lru_cache(maxsize=1)
def source_reader(path):
    # One reader per worker process; each member read verifies the ZIP CRC.
    return ZipFile(path)


def make_source(item, archive_path, output, conditions, fullscale=False):
    split,name,member = item
    raw = source_reader(archive_path).read(member)
    digest = sha(raw)
    with Image.open(io.BytesIO(raw)) as original:
        if original.mode!="L" or original.size!=(512,512):
            raise ValueError(f"Expected a grayscale 512×512 source: {name}")
        image = np.asarray(original.resize((256,256),Image.Resampling.BICUBIC),dtype=np.uint8)
    seed = int(sha(f"boss-hill-small-v1|{name}|{digest}|0.4".encode("ascii"))[:16],16)
    random_field = np.random.default_rng(seed).random(image.shape)
    costs = {}
    results = []
    for condition in (conditions if fullscale or split=="test" else ("HILL_04",)):
        outname = condition if fullscale else (f"test/{condition}" if split=="test" else "train")
        target = output/outname
        if condition=="JPEG_03":
            cover,stego = jpeg_pair(image,digest)
        else:
            method,rate = condition.split("_")
            if method not in costs:
                costs[method] = hill_cost(image) if method=="HILL" else wow_cost(image)
            cover,stego = image,spatial_draw(image,costs[method],random_field,int(rate)/10)
        row = {"source":name,"split":split,"source_sha256":digest}
        for role,pixels in (("cover",cover),("stego",stego)):
            row[role] = f"{role}/{name}"
            encoded = b"P5\n256 256\n255\n"+pixels.tobytes()
            (target/row[role]).write_bytes(encoded)
            row[role+"_sha256"] = sha(encoded)
        results.append((outname,row))
    return results


def prepare(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a new output directory: {args.out}")
    if args.workers<1 or len(set(args.conditions))!=len(args.conditions):
        raise ValueError("Specify at least one worker and distinct conditions")
    with ZipFile(args.archive) as archive:
        members = archive_members(archive)
        rows = private_split(args.selection,members) if args.selection else study_split(members,args.cohort)
        jobs = [(s,n,members[n]) for s,n in rows]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as scratch:
        stage = Path(scratch)/"prepared"
        outputs = {}
        if any(s!="test" for s,_ in rows):
            outputs["train"] = []
        if any(s=="test" for s,_ in rows):
            outputs.update({f"test/{c}":[] for c in args.conditions})
        for target in outputs:
            for role in ("cover","stego"):
                (stage/target/role).mkdir(parents=True)
        operation = partial(make_source,archive_path=args.archive.resolve(),output=stage,conditions=args.conditions)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            seen = set()
            for result in pool.map(operation,jobs,chunksize=1):
                source_hash = result[0][1]["source_sha256"]
                if source_hash in seen:
                    raise ValueError("Repeated source contents under different filenames")
                seen.add(source_hash)
                for key,row in result:
                    outputs[key].append(row)
        for key,rows in outputs.items():
            with (stage/key/"manifest.csv").open("w",encoding="utf8",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{key}/manifest.csv: {len(rows)} source pairs",flush=True)
        stage.rename(args.out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True,type=Path,help="Local official BOSSbase 1.01 ZIP")
    p.add_argument("--out",required=True,type=Path,help="New output directory")
    split_options = p.add_mutually_exclusive_group()
    split_options.add_argument("--selection",type=Path,help="Optional private CSV with source and split columns")
    split_options.add_argument("--cohort",choices=("original","fresh"),default="original",
                               help="Original train/validation/test split, or the independent 256-source test cohort")
    p.add_argument("--conditions",nargs="+",choices=CONDITIONS,default=list(CONDITIONS),
                   help="Test conditions (training always uses HILL 0.4)")
    p.add_argument("--workers",type=int,default=4)
    prepare(p.parse_args())
