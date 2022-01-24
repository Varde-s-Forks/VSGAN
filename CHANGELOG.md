# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2021-12-24

### Added

- Add support for ESRGAN+ models, Real-ESRGAN models (including 2x and 1x if pixel-shuffle was used),
  and A-ESRGAN models.
- Add support for Newer-New-arch in ESRGAN new-to-old state dict conversion.

### Changed

- Rework model/arch file system structure to /models, /models/blocks and /models/ESRGAN.
- Rework ESRGAN architecture as a singular class, with all ESRGAN-specific operation done within it.
- Move ESRGAN-specific blocks within ESRGAN.py.

### Removed

- Removed some unused blocks from RRDBNet.

### Fixed

- Ensure `clip` parameter of VSGAN is a VapourSynth VideoNode object (a clip).
- Move RGB clip check to the constructor of VSGAN rather than `run()`.

## [1.4.1] - 2021-12-21

### Added

- Created new sphinx documentation, replacing the old Jekyll documentation.
- Added HISTORY.md file for recording history (now CHANGELOG.md).

### Changed

- Reword some error/warning messages, now less opinionated and more concise.
- Some attributes have been renamed to be more ambiguous in the hopes more Model Architectures get
  supported in the future.

### Fixed

- Fix model chaining. It now gets the correct model and model scale values for each FrameEval call.
- Fixed the pytorch extra group to correctly be optional and correctly reference a dependency.
- Some type-hinting has been corrected.

## [1.4.0] - 2021-12-13

### Added

- Added support for all RGB formats including float.

### Changed

- Heavily improved main model execution code.
- Replace current chunk system with a seamless chunk system using overlap.
- Add self-chaining system, calls can be made directly after another.
- Made torch dependency optional and pointed directly to torch+cuda.
  This is due to conflicting kinds of torch installation methods.

### Removed

- Remove JetBrains `.idea` folder, added to gitignore.

### Fixed

- Only transpose C for RGB if it's 3-channels.

## [1.3.1] - 2021-10-25

### Fixed

- Fix type annotations on Python versions older than 3.9.
- Use Python version 3.9.x for Dist workflow as 3.10 is not yet supported.

## [1.3.0] - 2021-10-07

### Added

- Allow specification of the input array dimension order.
- Add Jekyll Documentation in `gh-pages` branch.
- Added a VSGAN Jupyter Notebook (Colab), with an Open in Colab Badge on the README.

### Changed

- Drop support for Python versions older than 3.6.2, due to bugs discovered in NumPy.
- Replace setup.py/setuptools with Poetry.
- Rename `cv2_imread` to `frame_to_np`, don't reverse to BGR as it's unnecessary.
- More efficiently write an array to a VapourSynth VideoFrame.
- Inherit output clip properties from input clip.
- Moved README's information to the docs.
- Reworked the CD GitHub Workflow to auto-create a GitHub Release and push to PyPI.

### Removed

- Remove the need for plane_count, now gets it from the input frame.
- Don't define the transposes, it's unnecessary.

### Fixed

- Fixed a bug with frame plane access on VapourSynth API 4.

## [1.2.1] - 2020-12-27

### Added

- Add ability to check what the last loaded model is via `VSGAN.model` attribute.

## [1.2.0] - 2020-12-27

### Added

- Added type-hinting across the code base as well as some doc-strings.

### Changed

- A heavy warning discouraging the use of your CPU as a PyTorch device was added. Ability to use
  your CPU was hidden but reading the warning explains how to do so.
- Reduced required VapourSynth version to 48 or newer.

### Removed

- Remove the conversion to RGB prior to model execution. RGB is required for the Model, but let
  the user decide how to convert to format, what algorithm, how to deal with matrix, and so on.
- Removed setuptools from dependencies.

### Fixed

- Add a check to ensure input clip is RGB, since auto conversion was removed.
- Add missing documentation on [1.1.0]'s changes to scale and such.

## [1.1.0] - 2020-10-20

### Added

- Added two GitHub Action workflows for CI/CD.

### Changed

- Moved the majority of documentation and info from the GitHub Wikis system to the README.

### Fixed

- Replace hardcoded `in_nc`, `out_nc`, `nf`, `nb`, and `scale` with values taken directly from the model state.
- Check that a model has been loaded before `execute` can be called.

## [1.0.8] - 2019-12-19

### Changed

- Change the RGB conversion check's kernel to `Spline36`.

## [1.0.7] - 2019-11-29

### Removed

- Removed the color-space conversion implemented in [1.0.3] as it can be a lossy operation.
  Let the user decide how/if to convert back to the original format. E.g., what algorithm,
  what matrix, and so on.

### Fixed

- Replaced unsafe assert in `RRDBNet` with an if and raise, as asserts may be removed when
  optimised as python byte code files.

## [1.0.6] - 2019-10-20

### Added

- Detect ESRGAN old/new arch models via archaic trial-and-error.

## [1.0.5] - 2019-10-20

### Changed

- Reworked code from Functional to Object-oriented Programming.
- Improve code readability, project starting to get serious.

## [1.0.4] - 2019-10-20

### Added

- Add ability to tile the input to reduce VRAM (does not hide seams).

## [1.0.3] - 2019-10-20

### Added

- VapourSynth to requirements.

### Changed

- Convert back to original color-space after applying the model.

## [1.0.2] - 2019-10-16

### Added

- Ability to select device via argument.

## [1.0.1] - 2019-10-15

### Added

- README file with some basic information.

### Changed

- Improved RGB conversion by using `mvsfunc` instead of `core.resize.Point`.

## [1.0.0] - 2019-09-21

Initial Release.

[Unreleased]: https://github.com/rlaphoenix/VSGAN/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.5.0
[1.4.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.4.1
[1.4.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.4.0
[1.3.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.3.1
[1.3.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.3.0
[1.2.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.2.1
[1.2.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.2.0
[1.1.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.1.0
[1.0.8]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.8
[1.0.7]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.7
[1.0.6]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.6
[1.0.5]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.5
[1.0.4]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.4
[1.0.3]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.3
[1.0.2]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.2
[1.0.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.1
[1.0.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.0