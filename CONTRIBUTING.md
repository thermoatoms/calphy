# Contributing to calphy

Thanks for your interest in calphy. Bug reports, fixes and new features are all
welcome.

## Licensing of contributions

Please read this before opening a pull request.

calphy is distributed under the **Academic Software Licence v1.0 (ASL)** — see
[LICENSE](LICENSE). The ASL is a reciprocal licence whose core terms match the GNU
GPLv2, but it permits **academic non-commercial use only**. It is an available-source
licence, not an open-source licence, and it is **not compatible with the GPL** or with
other copyleft open-source licences.

By submitting a contribution (a pull request, a patch, or a code suggestion in an
issue) you agree that:

1. Your contribution is licensed to the project and to its users under the same terms
   as calphy itself, i.e. the ASL (inbound = outbound).
2. You either own the copyright in your contribution, or you have the right from your
   employer or institution to submit it under those terms.
3. Your contribution does not include third-party code under a licence incompatible
   with the ASL. In particular, **do not** paste in code taken from GPL-, LGPL- or
   AGPL-licensed projects, including LAMMPS itself. Permissively licensed code
   (BSD, MIT, Apache-2.0) is acceptable provided you keep its copyright notice and
   say where it came from.

You keep the copyright in your contribution. Contributors are credited on the
[contributors page](https://github.com/ICAMS/calphy/graphs/contributors) and,
for larger contributions, in the acknowledgements section of the documentation.

If you cannot contribute under these terms, please open an issue describing the
change instead of submitting code, and we will find another way forward.

## Third-party dependencies

calphy depends only on permissively licensed Python packages. LAMMPS (GPLv2) is a
separate program: in the default `executable` execution mode calphy runs the `lmp`
binary as a subprocess and never links against it. The optional `library` execution
mode (`pip install calphy[library]`) loads LAMMPS in-process through `pylammpsmpi`;
that combination is made by the user at install time and calphy does not redistribute
LAMMPS.

Please do not add dependencies under copyleft licences without raising it first.

## Practical notes

- Run the test suite with `pytest` before opening a PR. Tests that need a real LAMMPS
  binary are marked `lammps`; the slower end-to-end ones are marked `slow`.
- New modules in `calphy/` should carry the standard licence header — copy it from any
  existing module in that package.
- Please keep pull requests focused; unrelated changes are easier to review separately.

## Questions

For anything not covered here, or to enquire about commercial use rights, contact
<sarath.menon@ruhr-uni-bochum.de>.
