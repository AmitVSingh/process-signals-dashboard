# Process Signals Dashboard
Link to dashboard: https://process-signals-dashboard-gbpedqd3uma7yxgjonfsfz.streamlit.app/
Interactive Streamlit dashboard for analyzing machine process signals from Excel files.


## Features
- Upload Excel files with machine-generated signals
- Automatic signal detection from headers
- Moving average smoothing (user-selectable)
- 3×3 analysis plot:
  - Time series (raw + MA)
  - Histogram
  - FFT magnitude
- Frequency polygon plots
- Interactive 3D Plotly visualization with configurable color indexing
- PNG export for all plots

## Expected Excel Format
Columns should follow the pattern:
- `Time - <signal name>`
- `<anything> - <signal name>`

CI smoke test: verifying GitHub Actions.
