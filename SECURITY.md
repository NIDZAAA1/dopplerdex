# Security policy

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability or misuse

Please **do not open a public issue**. Use GitHub's private vulnerability reporting:
*Security* tab of this repository, then *Report a vulnerability*. You will get an answer within a week.

Relevant reports include, for example:

- the local web app exposing files, recordings or results to other machines or users
  (it binds to `127.0.0.1` by default and should never be exposed publicly);
- path traversal or code execution through uploaded files or job parameters;
- a recording, private location or secret accidentally present in the repository or its history;
- dependencies with known vulnerabilities that affect dopplerdex.

## Running the web app safely

`dopplerdex-web` is a single-user local tool. Do not bind it to a public interface or put it on the
internet: it accepts uploads and runs long computations without authentication.
