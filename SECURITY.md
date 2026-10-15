# Security Policy

This document describes how to report security vulnerabilities in **Agentic Machine
Learning Engineering**, which versions receive fixes, and the project's security
posture. It is the single entry point for the security process for this repository.

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue, pull request, or commit that discloses a
security vulnerability.** Public disclosure before a fix is available puts all users at
risk.

If you discover a potential security issue in this project, we ask that you report it to
us privately through one of the following channels:

1. **AWS Vulnerability Reporting** — If you believe you have found a security issue in
   this AWS-published open-source project, notify AWS Security via
   [https://aws.amazon.com/security/vulnerability-reporting/](https://aws.amazon.com/security/vulnerability-reporting/)
   or email **aws-security@amazon.com**. Please do **not** create a public GitHub issue.
2. **GitHub private vulnerability reporting** — If enabled for this repository, use
   GitHub's "Report a vulnerability" button under the **Security** tab to open a private
   advisory with the maintainers.

When reporting, please include:

- The affected file/component (with `path:line` where possible).
- A description of the vulnerability and its potential impact.
- Reproduction steps or a minimal proof-of-concept.
- Any suggested remediation, if known.

We will acknowledge your report, work with you to understand and validate the issue, and
keep you informed of remediation progress. Please give us a reasonable opportunity to
address the issue before any public disclosure.

### Exposed credentials

If your report concerns **leaked credentials** (API keys, AWS secret keys, private keys),
treat it as urgent: the credential should be **rotated first**, then reported. Account IDs
and ARNs on their own are low-sensitivity (recon-only) and are handled as a hygiene fix,
not a credential-rotation emergency.

---

## Supported Versions

This project is under active development. Security fixes are applied to the latest
released version on the default branch. We recommend always running the most recent
release.

| Version | Supported |
|---------|-----------|
| `0.1.x` (latest) | ✅ |
| older pre-releases | ❌ |

---

## Security Posture

The surface is a pure-Python control plane that orchestrates AWS-managed primitives
(Bedrock, SageMaker, Glue, S3, CloudWatch) behind a ports-and-adapters seam. The full
offline mode runs with **no AWS credentials and no network egress to AWS**. The notes
below summarize the in-product controls relevant to security.

### Secrets handling

- **No secret material in code or git.** Runtime `.env*` files are gitignored; only
  placeholder `*.example` templates are tracked.
- **Externalized config.** Customer and adopter account/org-bearing values are never
  hardcoded; they resolve from environment variables, then SSM Parameter Store
  (`/amlas/*`), then a safe default
  ([`config/externalized.py`](src/amlas/config/externalized.py)). Required keys (e.g. the
  AWS account id) fail loudly rather than falling back to a hardcoded value. Reviewed
  AWS service-publisher account IDs are maintained in the public-release verifier.
- **Runtime secret resolution.** Secret *values* live in AWS Secrets Manager; only the
  secret *id* flows through config and the container image — never the value. Secrets are
  masked in logs and stripped from outbound HTTP error messages.

### In-band guardrails (safe-by-default)

- **Approval / permission gates.** Expensive or irreversible actions (launch a training
  job, deploy an endpoint, delete a resource) are classified and gated behind an explicit
  approval prompt ([`control_plane/approvals.py`](src/amlas/control_plane/approvals.py));
  cheap reads auto-approve.
- **Plan Mode.** A read-only preview with a cost/impact estimate runs before any spend
  ([`control_plane/plan_mode.py`](src/amlas/control_plane/plan_mode.py)).
- **Hooks-as-guardrails.** Deterministic `PreToolUse`/`PostToolUse` guardrails can *block*
  actions (e.g. cost-ceiling breaches, medallion bronze-write rejection, leakage checks)
  ([`control_plane/guardrails.py`](src/amlas/control_plane/guardrails.py)).
- **Headless safety.** Non-interactive runs deny gated actions unless explicitly
  allowlisted and capped by a max budget.

### Transport & storage encryption

- Bundled research providers use HTTPS. The anonymous supplied-URL fallback accepts
  HTTP(S) URLs, checks public addresses and redirect targets, and blocks private,
  loopback, link-local, and metadata addresses. HTTPS requests retain certificate
  and hostname verification.
- AWS SDK calls use TLS by default.
- CDK-provisioned S3 buckets enforce SSL, block public access, and enable
  server-side encryption + versioning; the DynamoDB jobs table uses AWS-managed encryption
  with point-in-time recovery ([`infra/cdk_stacks/`](infra/cdk_stacks)).

### Least-privilege IAM

- AWS access uses SigV4 via `boto3` sessions — no static-key or manual request signing.
- Per-backend least-privilege IAM stacks bound the blast radius of each adapter
  ([`infra/cdk_stacks/backend_iam_stack.py`](infra/cdk_stacks/backend_iam_stack.py)); the
  chat agent's filesystem tools are read-only, no-shell, and sandbox-confined.

### Supply chain

Dependencies are pinned in `uv.lock`. GitHub Actions and GitLab CI run the same
public-tree, lint, SAST, dependency, secret, test, and SBOM gates. Review dependency
changes before adoption, and regenerate the lock only after validating the resulting
package build.

---

## Disclaimer

This is sample/reference software provided under the [MIT No Attribution License (MIT-0)](LICENSE), published
following the **aws-samples** conventions. The managed-AWS backends provision real AWS resources
that **incur cost**; review the cost and approval guardrails before enabling live legs. You are
responsible for securing your own AWS account, credentials, and any data you process with this
software.

See the [README sample-code notice](README.md#agentic-ml-engineering) for deployment responsibilities.
