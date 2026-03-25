import sys
import os
from typing import Optional, Dict, Any

# Add project root to Python path (needed when running from different contexts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import httpx
from h2o_wave import main, app, Q, ui

# Try to import config, fallback to defaults
# Prefer API_BASE_URL env var (for client connections) over API_HOST (for server binding)
try:
    API_BASE_URL = os.getenv("API_BASE_URL")
    if not API_BASE_URL:
        from src.config import settings
        # API_HOST is 0.0.0.0 for server binding, not for client connections
        # Use localhost for local connections
        host = "localhost" if settings.API_HOST == "0.0.0.0" else settings.API_HOST
        API_BASE_URL = f"http://{host}:{settings.API_PORT}"
except ImportError:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# HTTP client for API calls
http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client for API calls."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            timeout=30.0
        )
    return http_client


async def check_api_health() -> Dict[str, Any]:
    try:
        client = await get_http_client()
        response = await client.get("/health")
        return {"status": "ok", "data": response.json()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def predict_via_api(transaction: Dict[str, Any]) -> Dict[str, Any]:
    try:
        client = await get_http_client()
        response = await client.post("/predict", json=transaction)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        # API returned error status
        error_detail = "API error"
        try:
            error_data = e.response.json()
            error_detail = error_data.get("message", error_data.get("detail", str(e)))
        except:
            error_detail = str(e)
        raise Exception(f"API Error: {error_detail}")
    except httpx.RequestError as e:
        # Connection error
        raise Exception(f"Cannot connect to API at {API_BASE_URL}. Is the API server running?")
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


@app('/fraud')
async def serve(q: Q):
    
    # Handle clear button first - before rendering anything
    if q.args.clear:
        # Clear all previous results
        del q.page['result_header']
        del q.page['decision']
        del q.page['shap_chart']
        del q.page['insights']
        del q.page['error']
        del q.page['api_status']
        # Reset form values in client state
        q.client.amount = ''
        q.client.card1 = '1000'
        q.client.transaction_id = '1'

    # Initialize the page with modern theme
    q.page['meta'] = ui.meta_card(
        box='',
        title='Financial Fraud Sentinel',
        theme='h2o-light',
        layouts=[
            ui.layout(
                breakpoint='xs',
                zones=[
                    ui.zone('header', size='80px'),
                    ui.zone('body', zones=[
                        ui.zone('sidebar', size='380px'),
                        ui.zone('main', zones=[
                            ui.zone('top'),
                            ui.zone('bottom')
                        ])
                    ])
                ]
            )
        ]
    )


    q.page['header'] = ui.header_card(
        box='header',
        title='Financial Fraud Sentinel',
        subtitle=f'AI-Powered Transaction Analysis' 
    )
    
    # Use client state for persistent values, with fallback to args
    if not q.args.clear:
        if q.args.amount is not None:
            q.client.amount = q.args.amount
        if q.args.card1 is not None:
            q.client.card1 = q.args.card1
        if q.args.transaction_id is not None:
            q.client.transaction_id = q.args.transaction_id

    # Enhanced input form with modern styling
    q.page['input_form'] = ui.form_card(
        box='sidebar',
        items=[
            ui.text_xl('**Transaction Analysis**'),
            ui.separator(),
            ui.textbox(
                name='amount',
                label='Transaction Amount ($)',
                value=q.client.amount if hasattr(q.client, 'amount') else '',
                placeholder='Enter amount (e.g., 150.00)',
                icon='Money',
                required=True
            ),
            ui.textbox(
                name='card1',
                label='Card ID',
                value=q.client.card1 if hasattr(q.client, 'card1') else '1000',
                placeholder='Enter card identifier',
                icon='CreditCardPerson'
            ),
            ui.textbox(
                name='transaction_id',
                label='Transaction ID',
                value=q.client.transaction_id if hasattr(q.client, 'transaction_id') else '1',
                placeholder='Enter transaction ID',
                icon='NumberSymbol'
            ),
            ui.separator(),
            ui.buttons([
                ui.button(
                    name='predict',
                    label='Analyze Transaction',
                    primary=True,
                    icon='PlayCircle'
                ),
                ui.button(
                    name='clear',
                    label='Clear',
                    icon='Clear'
                ),
                ui.button(
                    name='check_health',
                    label='Check API',
                    icon='Health'
                )
            ]),
        ]
    )
    
    # Handle API health check
    if q.args.check_health:
        health = await check_api_health()
        if health["status"] == "ok":
            health_data = health["data"]
            status_icon = '✅' if health_data.get("status") == "healthy" else '⚠️'
            q.page['api_status'] = ui.form_card(
                box='top',
                items=[
                    ui.message_bar(
                        type='success' if health_data.get("status") == "healthy" else 'warning',
                        text=f'{status_icon} API Status: {health_data.get("status", "unknown").upper()}'
                    ),
                    ui.text(f'Model Loaded: {"✓" if health_data.get("model_loaded") else "✗"}'),
                    ui.text(f'H2O Connected: {"✓" if health_data.get("h2o_connected") else "✗"}'),
                    ui.text(f'Uptime: {health_data.get("uptime_seconds", 0):.1f}s'),
                ]
            )
        else:
            q.page['api_status'] = ui.form_card(
                box='top',
                items=[
                    ui.message_bar(type='error', text=f'❌ API Health Check Failed'),
                    ui.text(f'Error: {health["message"]}'),
                    ui.text(f'Please ensure the API server is running at {API_BASE_URL}'),
                ]
            )
    
    # Show welcome/info screen initially or after clear
    if not q.args.predict or q.args.clear:
        if not q.args.check_health:  # Don't override health check display
            q.page['welcome'] = ui.form_card(
                box='top',
                items=[
                    ui.text_xl('**Welcome to Fraud Detection**'),
                    ui.separator(),
                    ui.text_l('Enter transaction details on the left to begin analysis'),
                    ui.separator(),
                    ui.text('This AI-powered system analyzes financial transactions in real-time to detect potential fraud using machine learning and explainable AI.'),
                    ui.separator(),
                    ui.text(f'**API Endpoint:** {API_BASE_URL}'),
                    ui.text('Click "Check API" to verify connectivity'),
                ]
            )

    # Process prediction (but not if clear was pressed)
    if q.args.predict and q.args.amount and not q.args.clear:
        try:
            # Prepare transaction data for API
            transaction = {
                "TransactionAmt": float(q.args.amount),
                "card1": int(q.args.card1 or 1000)
            }

            # Call API
            result = await predict_via_api(transaction)
            
            # Extract response data
            fraud_prob = result['fraud_probability']
            is_fraud = result['prediction'] == 1
            risk_level_value = result['risk_level']
            top_features_list = result.get('top_features', [])
            
            # Convert top features to dict format for compatibility
            top_features = {
                feat['feature']: feat['contribution']
                for feat in top_features_list
            }
            
            # Determine risk colors and icons
            if fraud_prob >= 0.7:
                risk_display = 'HIGH RISK'
                risk_color = '#FF6B6B'
                risk_icon = 'Warning'
            elif fraud_prob >= 0.3:
                risk_display = 'MEDIUM RISK'
                risk_color = '#FFD93D'
                risk_icon = 'Info'
            else:
                risk_display = 'LOW RISK'
                risk_color = '#6BCF7F'
                risk_icon = 'CheckMark'

            # Main result card with modern design
            q.page['result_header'] = ui.form_card(
                box='top',
                items=[
                    ui.text_xl(f'**Analysis Results**'),
                    ui.text(f'Request ID: {result.get("request_id", "N/A")}', size='s'),
                    ui.separator(),
                    ui.stats([
                        ui.stat(
                            label='Fraud Probability',
                            value=f'{fraud_prob:.1%}',
                            caption='Confidence score',
                            icon='BarChart4'
                        ),
                        ui.stat(
                            label='Risk Level',
                            value=risk_display,
                            caption='Based on probability',
                            icon=risk_icon
                        ),
                        ui.stat(
                            label='Transaction Amount',
                            value=f'${transaction["TransactionAmt"]:.2f}',
                            caption='Analyzed amount',
                            icon='Money'
                        ),
                    ]),
                    ui.separator(),
                    ui.progress(
                        label=f'Fraud Score: {fraud_prob:.1%}',
                        caption='Higher percentage indicates higher fraud risk',
                        value=fraud_prob
                    ),
                ]
            )
            
            # Decision card
            decision_text = '🚨 **FRAUD DETECTED**' if is_fraud else '✅ **LEGITIMATE TRANSACTION**'
            
            q.page['decision'] = ui.form_card(
                box='top',
                items=[
                    ui.message_bar(
                        type='error' if is_fraud else 'success',
                        text=decision_text
                    ),
                    ui.text(
                        f'The model predicts this transaction is **{"fraudulent" if is_fraud else "legitimate"}** '
                        f'with **{fraud_prob:.1%}** confidence.',
                        size='l'
                    ),
                ]
            )

            # Prepare SHAP data for plotting
            if top_features:
                shap_data = [
                    {
                        "feature": k,
                        "importance": abs(v),
                        "direction": "Increases Risk" if v > 0 else "Decreases Risk"
                    }
                    for k, v in top_features.items()
                ]

                # Enhanced SHAP visualization
                q.page['shap_chart'] = ui.plot_card(
                    box='bottom',
                    title='Top Contributing Features (SHAP Analysis)',
                    data=shap_data,
                    plot=ui.plot([
                        ui.mark(
                            type='interval',
                            x='=importance',
                            y='=feature',
                            color='=direction',
                            x_title='Impact on Fraud Prediction',
                            y_title='Feature'
                        )
                    ])
                )
            
            # Additional insights card
            if top_features:
                top_feature = list(top_features.keys())[0]
                top_importance = list(top_features.values())[0]
                
                q.page['insights'] = ui.form_card(
                    box='bottom',
                    items=[
                        ui.text_xl('**Key Insights**'),
                        ui.separator(),
                        ui.text_l(f'🔑 **Most Influential Feature:** {top_feature}'),
                        ui.text(f'Impact Score: {abs(top_importance):.4f}'),
                        ui.separator(),
                        ui.text('**Recommendation:**'),
                        ui.text(
                            '⚠️ Further investigation required. Review transaction details and contact customer.' 
                            if is_fraud else 
                            '✅ Transaction appears normal. Proceed with standard processing.',
                            size='m'
                        ),
                        ui.separator(),
                        ui.text(f'Model: {result.get("model_version", "Unknown")}', size='s'),
                    ]
                )
            
        except ValueError as e:
            q.page['error'] = ui.form_card(
                box='top',
                items=[
                    ui.message_bar(type='error', text=f'❌ Validation Error'),
                    ui.text(f'Error: {str(e)}'),
                    ui.text('Please enter valid numeric values for all fields.')
                ]
            )
        except Exception as e:
            q.page['error'] = ui.form_card(
                box='top',
                items=[
                    ui.message_bar(type='error', text=f'❌ Prediction Failed'),
                    ui.text(f'Error: {str(e)}'),
                    ui.separator(),
                    ui.text('**Troubleshooting:**'),
                    ui.text(f'1. Verify API is running at {API_BASE_URL}'),
                    ui.text('2. Click "Check API" to test connectivity'),
                    ui.text('3. Check API logs for detailed error information'),
                ]
            )

    await q.page.save()
