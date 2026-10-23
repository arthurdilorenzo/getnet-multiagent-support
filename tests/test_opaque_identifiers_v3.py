from app.security.pii import PIIRedactor


def test_bank_account_redaction_preserves_opaque_proposal_ids_and_hashes():
    identifier='PROP-0B5CC0654FBBD45EDB5A071A67670724F'
    digest='3804ce95e889bed8797381edfd108a9310fc3b602f15d110453b8fe0174d6d08'
    assert PIIRedactor.mask_text(identifier)==identifier
    assert PIIRedactor.mask_structure({'proposal_id':identifier,'payload_hash':digest})=={
        'proposal_id':identifier,'payload_hash':digest}
    assert '12345-6' not in PIIRedactor.mask_text('Conta corrente: 12345-6; CC 98765-4')
    assert '98765-4' not in PIIRedactor.mask_text('Conta corrente: 12345-6; CC 98765-4')
