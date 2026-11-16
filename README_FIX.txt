FIX: ModuleNotFoundError: No module named 'dashboard.renewable_plot'

Copy the included dashboard/renewable_plot.py into the existing project folder:
  C:\Users\hp\Documents\Data Engg Project\electricity_mlop_final\dashboard\renewable_plot.py

The file must be in the SAME dashboard folder as:
  monitoring.py
  ui.py
  anomaly_strips.py

Do not put renewable_plot.py inside dashboard/pages/.
Do not rename it.

Then stop Streamlit (Ctrl+C in the terminal) and restart from the project root:
  python -m streamlit run dashboard/app.py

The module uses pandas and Plotly, which are already project dependencies.
