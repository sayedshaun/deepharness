# Security policy

## Supported versions

deepharness is before 1.0, so only the latest release gets security fixes. Fixes land on `main`
first and ship in the next release.

| Version | Supported |
| --- | --- |
| 0.3.x | Yes |
| < 0.3 | No |

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately through GitHub:
on the repository's **Security** tab, choose **Report a vulnerability**. The report and the
discussion stay private until a fix is released.

A useful report includes:

- what an attacker can do, and what they need to control to do it (a prompt, a file the model
  reads, a tool's arguments, the host configuration);
- the deepharness version, Python version and operating system, and for the sandbox, the
  `bwrap` and systemd versions;
- the smallest code that shows the problem.

## What counts

These are in scope:

- a command run through `Bubblewrap` reaching what it shouldn't: writing outside the workspace,
  reading a home directory or the environment, using the network with `network=False`,
  outliving its timeout, or going past a memory or process limit;
- a sandbox or limit that silently doesn't apply when the documentation says it would refuse
  to start instead;
- `Workspace.resolve()` returning a path outside the root;
- `Permissions` or `requires_approval` letting through a call that a rule should have denied
  or paused, other than by getting around a glob pattern;
- the library itself leaking secrets, for example into logs.

These are not, because the [threat model](https://sayedshaun.github.io/deepharness/guide/threat-model/)
already describes them as the application's to handle:

- prompt injection in general;
- tools, MCP servers or other code you wrote or chose to run;
- the documented limits: deny rules that match by glob, the workspace and system directories
  being readable from the sandbox, `network=True` meaning the whole network, and no sandbox on
  platforms other than Linux;
- vulnerabilities in `bwrap`, systemd, the Linux kernel or a model provider, which belong with
  those projects.

If you're unsure whether something counts, report it privately anyway.
