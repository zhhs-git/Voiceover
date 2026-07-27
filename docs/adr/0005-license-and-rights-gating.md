# ADR 0005: License and Rights Gating

## Status

Accepted.

## Context

The app converts books into audiobooks. This may involve reproduction, transformation, and possibly derivative-work concerns depending on jurisdiction, book status, and intended use.

The app cannot reliably determine legal rights for every uploaded file.

## Decision

Implement best-effort license and rights checks plus user attestation.

The app should inspect:

- EPUB rights metadata
- PDF metadata
- visible license notices
- Creative Commons markers
- Project Gutenberg indicators
- publication year and author metadata
- DRM indicators

The app should classify uploads as:

- allowed
- restricted
- unknown
- blocked

For unknown or restricted works, the app should ask the user to confirm they have the right to convert the book for their intended use. The app should store the detected metadata, classification, attestation, and timestamp.

The app should not present the classification as legal advice.

## Consequences

Benefits:

- Reduces accidental misuse.
- Supports public-domain and permissively licensed workflows.
- Creates an audit trail for user decisions.
- Avoids pretending the app can solve copyright automatically.

Costs:

- Some lawful uses may still be flagged as unknown.
- Some metadata may be missing or misleading.
- Jurisdiction-specific rules remain outside the app's automated certainty.

## Alternatives Considered

No license gate:

- Simpler.
- Higher risk of misuse and user confusion.

Hard block all copyrighted-looking files:

- Conservative.
- Blocks legitimate personal, educational, accessibility, or licensed uses.

Automated legal determination:

- Not realistic.
- Too jurisdiction-dependent and metadata-dependent.
