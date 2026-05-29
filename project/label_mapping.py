# label_mapping.py

"""
Variable Star Label Mapping

This module converts detailed variable star classifications
from catalogs such as GCVS and ASAS-SN into a smaller set
of research labels.

Why is this necessary?

The original catalogs contain many detailed subclasses.
For example:

    RRAB
    RRC
    RRD
    DSCT
    HADS
    EA
    EB
    EW
    SRA
    SRB
    ...

Using every detailed class directly can cause problems:

- Too many labels
- Severe class imbalance
- Fewer samples per class
- Lower model stability

To simplify the classification task, detailed catalog labels
are grouped into six major classes:

    RR_Lyrae
    Cepheid
    Eclipsing
    Mira
    SemiRegular
    DeltaScuti
"""


def normalize_var_type(raw_type):
    """
    Normalize a variable star type string.

    Purpose
    -------
    Catalog labels often contain extra spaces,
    punctuation, or formatting symbols.

    Examples
    --------
    RRAB     -> RRAB
    M:       -> M
    EA/RS    -> EA/RS
    dsct     -> DSCT

    Parameters
    ----------
    raw_type : str

    Returns
    -------
    str
        Cleaned variable type string.
    """

    if raw_type is None:
        return ""

    text = str(raw_type).strip().upper()

    text = text.replace(":", "")
    text = text.replace(" ", "")

    return text


def map_to_group_label(raw_type):
    """
    Convert a detailed catalog variable star type
    into one of the six research labels.

    Output Labels
    -------------
    RR_Lyrae
    Cepheid
    Eclipsing
    Mira
    SemiRegular
    DeltaScuti

    Unsupported labels return None.

    Parameters
    ----------
    raw_type : str

    Returns
    -------
    str or None
    """

    t = normalize_var_type(raw_type)

    if not t:
        return None

    # --------------------------------------------------------
    # Composite labels
    #
    # Example:
    # EA/RS -> EA
    # DSCT+EA -> DSCT
    #
    # Use the primary classification only.
    # --------------------------------------------------------

    primary = t.split("/")[0]
    primary = primary.split("+")[0]

    # --------------------------------------------------------
    # RR Lyrae
    # --------------------------------------------------------

    if primary.startswith("RR"):
        return "RR_Lyrae"

    # --------------------------------------------------------
    # Cepheid
    # --------------------------------------------------------

    if (
        primary.startswith("DCEP")
        or primary.startswith("CEP")
        or primary.startswith("CW")
        or primary.startswith("T2CEP")
    ):
        return "Cepheid"

    # --------------------------------------------------------
    # Eclipsing Binary
    # --------------------------------------------------------

    if (
        primary == "E"
        or primary.startswith("EA")
        or primary.startswith("EB")
        or primary.startswith("EW")
        or primary.startswith("ELL")
    ):
        return "Eclipsing"

    # --------------------------------------------------------
    # Mira Variables
    # --------------------------------------------------------

    if primary == "M":
        return "Mira"

    # --------------------------------------------------------
    # Semi-Regular Variables
    # --------------------------------------------------------

    if primary.startswith("SR"):
        return "SemiRegular"

    # --------------------------------------------------------
    # Delta Scuti
    # --------------------------------------------------------

    if (
        primary.startswith("DSCT")
        or primary.startswith("HADS")
    ):
        return "DeltaScuti"

    # --------------------------------------------------------
    # Unsupported Type
    # --------------------------------------------------------

    return None