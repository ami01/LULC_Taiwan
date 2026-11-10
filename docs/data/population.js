// Taiwan total population, aligned with the LULC time steps.
// Historical: year-end household-registration population, Ministry of the Interior (MOI),
//   as tabulated on https://en.wikipedia.org/wiki/Demographics_of_Taiwan ("Historical Populations").
//   2010 cross-checked against MOI "Population Policy Data Collection (2020)", Table 3-2-2-1 (23,162,123).
// Projected: National Development Council (NDC), "Population Projections for the R.O.C. (Taiwan)",
//   medium variant — NOT YET FILLED IN. Enter the 2030 and 2050 values from
//   https://pop-proj.ndc.gov.tw/main_en/ and set `projectionSource` to the edition used.
window.POPULATION = {
  source: "Ministry of the Interior (Taiwan), year-end household registration",
  projectionSource: null, // e.g. "NDC Population Projections 2024–2070, medium variant"
  values: {
    1990: 20401305,
    2000: 22276672,
    2010: 23162123,
    2020: 23561236,
    2030: null,
    2050: null
  }
};
