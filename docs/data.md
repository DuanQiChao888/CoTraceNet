# Rebuilding the detection benchmarks

Download the original [BOSSbase 1.01 archive](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip) from Binghamton University's DDE Lab. Keep the ZIP intact. The sampler uses the original order of the archive members.

From the repository root, install the preparation dependencies in a separate Python environment and choose a new output directory.

```sh
python -m pip install -r tools/requirements-data.txt
python tools/prepare_pairs.py --archive /path/to/BOSSbase_1.01.zip --out /path/to/prepared --workers 4
```

The script writes binary 256 × 256 grayscale PGM images. Each manifest holds one row per source with relative paths to its cover and stego image, its split, and SHA-256 checksums. Split by source so both members of a pair stay together.

| Manifest under the output directory | Source pairs | Condition |
|---|---:|---|
| `train/manifest.csv` | 160 train + 48 validation | HILL, 0.4 bpp |
| `test/HILL_04/manifest.csv` | 256 | HILL, 0.4 bpp |
| `test/HILL_01/manifest.csv` | 256 | HILL, 0.1 bpp |
| `test/WOW_04/manifest.csv` | 256 | WOW, 0.4 bpp |
| `test/WOW_01/manifest.csv` | 256 | WOW, 0.1 bpp |
| `test/JPEG_03/manifest.csv` | 256 | J-UNIWARD, QF 85, 0.3 bpnzAC |

Release verification reproduced the source identity and order of all 464 training, validation and test photographs and compared every generated PGM against the historical experiment by SHA-256. All 2,976 image files matched, including the decoded JPEG condition.

### Source selection

The initial 512-image pool uses 64 equal strata in physical ZIP order. In each stratum, Python's seeded generator chooses eight adjacent entries. The historical 60/20/20 partition uses seed 17 on numerically sorted filenames. The final training and validation sources are sampled from their respective partitions using NumPy PCG64 and seeds derived from `[20261122, split_index]`. For the fourth evaluation cohort, PCG64 seed 20261126 draws 256 sources after excluding the initial pool and three earlier cohorts. All five evaluation conditions use this same fourth cohort.

To prepare your own split, pass `--selection /path/to/private.csv`. Supply columns `source` and `split`, where each source is an archive member's basename and the split is `train`, `val` or `test`. To generate only the principal evaluation condition, pass `--conditions HILL_04`.

### Image generation

- All source images are resized from 512 × 512 to 256 × 256 using Pillow bicubic interpolation before embedding.
- HILL and WOW use ideal independent ternary modifications with an entropy tolerance of 0.00001 bpp. A per-source hash seeds the same random field across spatial conditions. The preparation code computes spatial costs and samples evaluation perturbations. For WOW it obtains the Daubechies filter taps from [conSeal](https://github.com/uibk-uncover/conseal).
- J-UNIWARD compresses each resized image at quality 85, simulates the original J-UNIWARD method on quantized DCT coefficients, writes the resulting coefficients back to a JPEG, and decodes the cover and stego image with the same libjpeg 6b backend. Payload is measured per nonzero cover AC coefficient. The JPEG round trip uses [jpeglib](https://github.com/martinbenes1996/jpeglib) and [conSeal](https://github.com/uibk-uncover/conseal).
