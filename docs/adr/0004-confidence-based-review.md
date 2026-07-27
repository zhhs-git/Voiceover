# ADR 0004: Confidence-Based Review

## Status

Accepted.

## Context

Dialogue speaker attribution, gender inference, OCR quality assessment, and emotion tagging are imperfect. A fully manual review process would make audiobook generation tedious. A fully automatic process can produce obvious mistakes without giving the user a chance to correct high-impact errors.

The product goal is to minimize user intervention.

## Decision

Use confidence-based review.

The app should generate automatically when confidence is acceptable. It should flag low-confidence issues and offer optional correction, but it should not require line-by-line review unless generation cannot proceed.

When confidence is low, the app should use conservative fallbacks:

- unknown speaker uses narrator or neutral dialogue voice
- unknown gender uses neutral voice
- unknown emotion uses neutral emotion
- uncertain chapter boundary is flagged but still generated when possible

User corrections should apply globally where safe, such as alias merges, gender updates, and voice assignment changes.

## Consequences

Benefits:

- Keeps the default workflow fast.
- Avoids requiring the user to babysit every chapter.
- Provides a path to improve bad outputs.
- Helps the app learn from minimal corrections.

Costs:

- Confidence scoring must be designed and tuned.
- Some errors will still slip through.
- The UI must explain uncertainty without overwhelming the user.

## Alternatives Considered

Full manual review:

- Highest control.
- Poor user experience for long books.

Fully automatic generation:

- Fastest workflow.
- More likely to produce wrong voices or bad speaker attribution.

Block on uncertainty:

- Avoids some bad output.
- Frustrating for ambiguous novels and messy PDFs.
