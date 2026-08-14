# Data Policy

## Rule

Every training source must have known provenance and a documented right-to-use basis before ingestion.

## Never commit

- raw copyrighted corpora without redistribution rights;
- secrets or personal data;
- scraped private data;
- proprietary model outputs used as training data.

## Dataset record

Each source should eventually record:

- name;
- origin;
- license/usage basis;
- retrieval date;
- checksum;
- language/domain;
- filtering steps.

Raw training data stays outside Git unless explicitly redistributable and intentionally approved by an Issue.
