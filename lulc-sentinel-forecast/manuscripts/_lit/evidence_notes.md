# Evidence notes for references without an API abstract

Collected by web search (abstract pages, repository records) on 2026-10-29.
Use these, together with abstracts.json, as the `evidence` field of ledger rows.

- **Drusch et al. 2012, RSE 120:25-36 (10.1016/j.rse.2011.11.026)** - Sentinel-2
  mission paper; MultiSpectral Instrument with 13 spectral bands from visible and
  NIR to SWIR; mission provides high-revisit systematic high-resolution
  multispectral imagery and continuity with SPOT/Landsat-type data. (Swath and
  revisit not confirmed from the abstract - do not cite Drusch for those numbers.)
- **Olofsson et al. 2014, RSE 148:42-57 (10.1016/j.rse.2014.02.015)** - good
  practice recommendations organised in three components: sampling design,
  response design, analysis. Recommends a probability sampling design, reference
  labels more accurate than the map, error matrix reported as proportions of
  area with overall, user's and producer's accuracy, and class areas estimated
  from the reference labels.
- **van Vliet et al. 2016, EMS 82:174 (10.1016/j.envsoft.2016.04.017)** - review
  of calibration/validation in land-change modelling; most validation used
  locational accuracy; 31 % of reviewed applications reported no validation;
  argues validation should account for uncertainty, non-stationarity,
  equifinality.
- **Pontius et al. 2008 (online 2007), Ann Reg Sci 42:11-37
  (10.1007/s00168-007-0138-2)** - three-map comparison (initial reference, later
  reference, prediction) of 13 applications of 9 land-change models; error
  exceeded correctly predicted change in 12 of 13 applications at the raw
  resolution; figure of merit and null resolution as summary statistics; more
  accurate applications have larger observed net change.
- **Pontius, Huffaker & Denman 2004, Ecol Model 179:445 (10.1016/j.ecolmodel.2004.05.010)**
  - TLDR: introduces the null resolution, the resolution at which the model is as
  accurate as a null (no-change) model; applied to Geomod.
- **Takada, Miyamoto & Hasegawa 2010 (online 2009), Landscape Ecol 25:561-572
  (10.1007/s10980-009-9433-x)** - transition matrices measure land-use change
  rates, but observation intervals differ, so matrices must be normalised to a
  common (yearly) interval before comparison; the power-root formula used for
  this can have difficulties, including more than one yearly matrix fitting the
  same data.
- **Verburg & Overmars 2009, Landscape Ecol (10.1007/s10980-009-9355-7)** -
  Dyna-CLUE: combines demand-driven (top-down) allocation of land area with
  locally determined (bottom-up) conversion processes; applied to European
  farmland abandonment and vegetation regrowth.
- **Kamusoko et al. 2009, Applied Geography 29:435-447
  (10.1016/j.apgeog.2008.10.002)** - Markov-cellular automata model: Markov
  transition probabilities from satellite maps + CA spatial filter;
  multicriteria evaluation for transition potential maps; simulated to 2030 in
  Bindura district, Zimbabwe; projects woodland loss.
- **Guan et al. 2011** - abstract NOT verified; do not cite.
- **Mishra & Rai 2018, Applied Geomatics (10.1007/s12518-018-0223-5)** - TLDR +
  web search: Varanasi district; compared stochastic Markov, CA-Markov and
  MLP-Markov, calibrated on 1988 and 2001 maps to predict 2015; MLP-Markov gave
  the most reliable results; projections to 2030/2050 show increasing built-up
  and decreasing agricultural land.
- **Singh et al. 2015, Environmental Processes (10.1007/s40710-015-0062-x)** -
  web search: Allahabad (Prayagraj) district; CA-Markov calibrated on 1990 and
  2000, validated against 2010, projected 2020.
- **Singh et al. 2024, Sustainability 16:10722 (10.3390/su162310722)** - Hindon
  basin (western UP); Landsat 2002, 2013, 2023; CA-Markov to 2050; agricultural
  land from 60.98 % (2002) to 54.70 % (2050).
