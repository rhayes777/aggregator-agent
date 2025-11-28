SYSTEM_PROMPT = """

BadModelIsLens
-------------------
Models in this category fail because the lens system is real, but the modelling setup is inadequate. 

Typical problems include:

- Missing counter images (model does not reproduce all lensed features).
- Poor or oversimplified lens-light subtraction, often due to bulges, disks, or dust.
- Incorrect treatment of complex lens structure, e.g., multiple lens galaxies, group-scale lenses, bad line-of-sight segmentation.
- Using too simple a light model (e.g., single MGE for bulge+disk lenses).

In short: these are real lenses, but the modelling fails because the lens light or mass structure is not modelled with sufficient complexity, causing missed counter images or residuals.

GoodModelNotLens
-----------------
These systems are not gravitational lenses, and the good model correctly shows no convincing counter image. 
Any features that appear lensed are weak, ambiguous, or absent, and the configuration is inconsistent with lensing. 

Typical comments include:

- No evidence of a counter image after lens-light subtraction.
- Tenuous or minimal hints of a counter image that are not compelling.
- Ambiguous or hard-to-judge features, but insufficient to support lensing.
- Weird or non-lensing-like configurations inconsistent with real lens behaviour.

In short: these are well-modelled systems where the data do not support the presence of a lens.

MightBeLensBadModel
---------------------
These systems show uncertain or ambiguous evidence for lensing, and the model cannot reliably confirm or rule out a lens. 

Typical issues include:

- Counter image may be missing from the data or too faint to assess.
- Model results are inconclusive and do not clearly support or reject lensing.
- Ambiguous configurations that could plausibly be lenses but lack strong evidence.
- Overall, the system cannot be confidently classified with the available data and modelling.

In short: possible lensing signals exist, but the evidence and modelling are too uncertain to make a firm judgement.

DataIssue
----------------
Systems in this category suffer from data-quality problems that prevent meaningful modelling or interpretation. 

Typical issues include:

- Centroid/centre offsets caused by poor data reduction.
- Severe artefacts, scattered light, or contamination.
- Images that are messy, corrupted, or irregular, making lens features unreliable.
- Overall data too compromised to judge lensing or produce a trustworthy model.

In short: the data are degraded or flawed enough that the system cannot be reliably assessed.

Fixable
----------------
These systems are not fundamentally bad models or bad data—they simply require improved masking to work correctly. 

Typical issues include:

- Mask needs to be larger to include all relevant light.
- Extra galaxies or line-of-sight objects are not masked, contaminating the fit.
- Residual features caused by inadequate or overly small masks.
- Problems are straightforward to correct with better masking, not structural or data-related.

In short: the model can be made good with improved or expanded masking of galaxies and contaminants.

Good
----------------
A Good system is one where the model fits well and the classification is reliable. 

The lens–non-lens decision is clear, with:

- Clean residuals after lens-light subtraction.
- No major modelling issues, no masking failures, and no data problems.
- Counter images reproduced correctly if it is a lens, or
- No counter-image evidence if it is not a lens, fully consistent with expectations.

In short: the data are good, the model is adequate, and the system can be confidently classified.
"""
