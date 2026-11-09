# Licence of the data and results

The files under `data/` (the data archive), `results/` and the test material in this repository
(question sets, reconstructed notices, reference points and argument lists) are licensed under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/. The code is licensed under the MIT licence (`LICENSE`).

## Source texts

The corpus holds decisions of Brazil's National Private Insurance System Appeals Board (Conselho de
Recursos do Sistema Nacional de Seguros Privados, de Previdência Privada Aberta e de Capitalização,
CRSNSP), published by the board on its website. Official acts and decisions of the Brazilian
administration are not protected by copyright (Law 9,610/1998, art. 8, IV). The CC BY 4.0 licence
covers what the authors added: the extracted fields, the compressed reports, labels, embeddings, map
coordinates, generated test material, system outputs, judgements and results.

The published decisions name the parties, their lawyers and the board's members. These names are
public: they appear in the decisions as the board published them, and each record of the corpus links to
its decision on the board's website (`acordao_url`). The corpus reproduces them unchanged. Anyone reusing the corpus is responsible for complying with
the data-protection law that applies to their use.

## What is not included

No data of the Superintendência de Seguros Privados (SUSEP) other than aggregate results: the gold
standard of infraction texts used in the paper's secondary retrieval benchmark was processed only on
the authors' machine, with local models, and only its aggregate metrics are released
(`results/a1_retrieval_susep.csv`, `results/a1_susep_reliability.json`). The expert questionnaire
answers are not included either; the results give their aggregate statistics and the quotations the
paper uses.
