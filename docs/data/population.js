// Taiwan total population, aligned with the LULC time steps.
// Historical: year-end household-registration population, Ministry of the Interior (MOI),
//   as tabulated on https://en.wikipedia.org/wiki/Demographics_of_Taiwan ("Historical Populations").
//   2010 cross-checked against MOI "Population Policy Data Collection (2020)", Table 3-2-2-1 (23,162,123).
// Projected: National Development Council (NDC), "Population Projections for the R.O.C. (Taiwan):
//   2024–2070", medium variant. The report does not print 2030/2050 totals, so they are derived from
//   its medium-variant age tables (p.12): total = group count ÷ group share, using both the 15–64 and
//   65+ groups (counts in 10,000 persons, shares in %):
//     2030: 1,509 ÷ 65.8% = 2,293 ; 556 ÷ 24.2% = 2,298  → ~2,293 (rounding range 2,291–2,296)
//     2050: 1,060 ÷ 53.8% = 1,970 ; 757 ÷ 38.4% = 1,971  → ~1,970 (rounding range 1,968–1,973)
//   Method check: the same calculation for 2024 gives 2,340, matching the report's printed total.
//   Values rounded to the nearest 10,000.
window.POPULATION = {
  source: "Ministry of the Interior (Taiwan), year-end household registration",
  projectionSource: "NDC Population Projections 2024–2070, medium variant (totals derived from age tables)",
  values: {
    1990: 20401305,
    2000: 22276672,
    2010: 23162123,
    2020: 23561236,
    2030: 22930000,
    2050: 19700000
  },
  // Population figures stated in the source paper (Liu & Arora, "Decadal analysis and simulation of
  // land use and land cover changes…", Sect. 2 Study area & Sect. 5). The paper gives no per-year values;
  // its population projections are only in Supplementary Fig. 1.
  paper: {
    growth_1990_2020_pct: 15.6,   // "from 1990–2020, the population growth is recorded 15.6%"
    total_millions: 23,           // "Taiwan's total population of 23 million"
    trend_note: "Negative population growth since 2020"
  }
};
