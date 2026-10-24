# Video Gallery

The [repository front page](../../README.md#videos-and-mp4-demonstrations) features eight retained simulation movies. Download the originals from the [video release](https://github.com/rbajaj5/delay-aware-runtime-assurance-lab/releases/tag/videos-2026-10-24).

## Scope

These are exploratory simulations and teaching illustrations, not real-world flight tests, real basketball data, or evidence of betting profitability. The infrastructure movie records the local planning dashboard, using observed Washington demand and synthetic/model-based scenarios with the limitations documented in the experiment. The bridge-fleet prototype is a separate earlier schematic experiment.

The portfolio movie's final quadrotor glyphs preserve the basketball state to illustrate a mathematical analogy. They are not a second robotics experiment. GPU video encoding must not be confused with GPU physics: the planning and portfolio simulations run on the CPU; selected formation experiments separately used CUDA.

These movies accompany the simulation lab. Links from the RAG repository do not imply that the RAG generated, controlled, or independently validated the simulations.

## Integrity and Reproduction

`video_catalog.json` records the original source path, exact asset name, byte size, SHA-256, duration, codec, dimensions, audio/subtitle streams, preview timestamp, scope, and associated methods/report for each movie. Burned-in captions are not counted as a subtitle stream.

The October 24 media check decoded every video/audio frame with FFmpeg and extracted each preview directly from its corresponding MP4. It did not modify the movies, rerun experiments, or revalidate their scientific claims. Historical reports describe the original runs. The prior source-release manifest remains an immutable snapshot of that earlier commit; this catalog covers the new media assets.

To regenerate metadata and previews after downloading all eight release MP4s into a directory, install FFmpeg and run from the repository root:

```bash
python analysis/prepare_public_video_gallery.py --asset-dir /path/to/downloaded/mp4s
```

With the complete original local workspace, omit `--asset-dir`. The script reads the retained movies, validates decoding, and regenerates JPGs and the JSON catalog. It makes no network calls and does not publish. To regenerate the actual simulations, consult each entry's report and generator; some experiments require large inputs or traces excluded from the source release.

The GitHub README uses linked preview images rather than unsupported inline video HTML. Depending on the browser, an MP4 link may download instead of opening a player. The release supplies the original playable MP4 files in either case.
