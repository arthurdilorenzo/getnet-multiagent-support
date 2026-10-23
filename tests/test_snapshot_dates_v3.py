from app.agents.customer_support_agent import CustomerSupportAgent
from app.security.auth import SecurityContext
from app.tools.customer_tools import get_sales_settlement


def test_snapshot_does_not_confirm_a_current_payout():
    output=CustomerSupportAgent().run('Quando caem as vendas de ontem?',SecurityContext(authenticated_user_id='cliente1988'))
    assert 'snapshot' in output.response.lower()
    assert 'não confirma' in output.response.lower()
    assert '**Valor Líquido Programado para Hoje:**' not in output.response
    assert 'liquidação hoje' not in output.response.lower()


def test_explicit_sales_date_does_not_return_another_days_money():
    result=get_sales_settlement('cliente1988',sales_date='2040-01-01')
    assert result['found'] is False
    assert result['available_snapshot_date']=='2026-10-21'
    assert not result.get('sales_data')
