# Dataset source and attribution

Dataset: SMS Spam Collection
Creators listed by UCI: Tiago Almeida and Jose Maria Gomez Hidalgo
DOI: https://doi.org/10.24432/C5CC84
Repository: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Download: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
License listed by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0)
License text: https://creativecommons.org/licenses/by/4.0/legalcode
Fetched for this MVP on 2026-10-10.

Downloaded archive SHA-256:
1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3

Original `SMSSpamCollection` SHA-256:
7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d

Verified original rows: 5574 (4827 ham and 747 spam).
The raw file has 5171 distinct exact label/message records and 5159 distinct case-folded, whitespace-normalised message keys. The model's preparation report records exclusions and deduplication before splitting.

The original `SMSSpamCollection` and `readme` remain unchanged. Training preparation excludes conflicting-label keys, removes normalised message duplicates and creates disjoint training/test subsets. The attack only changes a COPY of untrusted training labels; all such exports are named and described as experiment data.

Introductory paper: T. A. Almeida, J. M. G. Hidalgo and A. Yamakami (2011), Contributions to the study of SMS spam filtering: new collection and results. https://doi.org/10.1145/2034691.2034742

This source is an older spam research corpus, not a current phishing dataset.
