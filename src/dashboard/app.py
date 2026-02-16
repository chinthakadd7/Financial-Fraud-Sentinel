import sys
import os

# Add project root to Python path (needed when running from different contexts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from h2o_wave import main, app, Q, ui
from src.models.predictor import predict_transaction


@app('/fraud')
async def serve(q: Q):

    if not q.client.initialized:
        q.client.initialized = True
        q.page['meta'] = ui.meta_card(box='', title='Financial Fraud Sentinel')

    q.page['header'] = ui.header_card(
        box='1 1 4 1',
        title='🚨 Financial Fraud Sentinel',
        subtitle='H2O Wave + Explainable AI'
    )

    q.page['form'] = ui.form_card(
        box='1 2 4 4',
        items=[
            ui.textbox(name='amount', label='Transaction Amount'),
            ui.button(name='predict', label='Predict Fraud', primary=True)
        ]
    )

    if q.args.predict:
        transaction = {
            "TransactionAmt": float(q.args.amount),
            "TransactionID": 1,
            "card1": 1000
        }

        result = predict_transaction(transaction)

        q.page['result'] = ui.form_card(
            box='1 6 4 2',
            items=[
                ui.text(f"Fraud Probability: {result['fraud_probability']:.4f}"),
                ui.text(f"Prediction: {'Fraud' if result['prediction'] == 1 else 'Not Fraud'}")
            ]
        )

        # Prepare SHAP data for plotting
        shap_data = [
            {"feature": k, "importance": v}
            for k, v in result["top_features"].items()
        ]

        q.page['shap_chart'] = ui.plot_card(
            box='5 2 4 6',
            title='Top 3 Fraud Drivers (SHAP)',
            data=shap_data,
            plot=ui.plot([
                ui.mark(type='interval', x='=feature', y='=importance')
            ])
        )

    await q.page.save()
