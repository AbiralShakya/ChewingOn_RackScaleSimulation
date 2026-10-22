# Security policy

rackscale is a simulator that reads YAML, JSON and recorded traces and writes reports. The parts
that touch untrusted input are the YAML/JSON loaders, the capture importers, and `rackscale
capture`, which runs the program you give it under NVBit.

Please report a vulnerability privately through GitHub's
[private vulnerability reporting](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/security/advisories/new)
rather than a public issue. We aim to acknowledge reports within a week.

Only the latest release and `main` receive fixes.
