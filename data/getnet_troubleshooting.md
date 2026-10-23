---
source_url: https://site.getnet.com.br/duvidas/
secondary_url: https://site.getnet.com.br/get-ajuda/
canonical_domain: site.getnet.com.br
title: Suporte Técnico Oficial Getnet — Conectividade de Maquininhas e Erros de Transação
---

# Suporte Técnico e Diagnóstico Operacional de Maquininhas Getnet

Fonte oficial: `https://site.getnet.com.br/duvidas/` e `https://site.getnet.com.br/get-ajuda/`

## O que fazer quando a maquininha não conecta na internet (Sem Sinal / Erro de Rede)?
Para restabelecer a comunicação de rede (Chip multi-operadora Algar/Claro/Vivo/Tim ou Wi-Fi) em terminais Get Clássica ou Get Smart, execute o protocolo técnico:

1. **Verificação e Reinserção do SIM Card (Procedimento Oficial Getnet):**
   - Com a máquina totalmente desligada, abra a tampa na parte traseira.
   - Retire e insira o chip no slot **SIM1** ou **SIM2**, garantindo que a parte metálica dourada esteja voltada para baixo e devidamente encaixada.
   - Os terminais contam com tecnologia de **Chip Multi-Operadora (ex.: Chip Algar)**, que busca automaticamente a operadora com melhor cobertura 2G/3G/4G na região.
2. **Reinicialização do Equipamento (Ciclo de Energia):**
   - Desligue a maquininha pelo botão de energia/cancelar (vermelho), aguarde 10 segundos e ligue novamente para forçar nova autenticação na torre de telefonia móvel.
3. **Alternância entre Rede Móvel (GPRS/3G/4G) e Wi-Fi:**
   - Acesse as configurações de rede do terminal e conecte a uma rede Wi-Fi local estável, ou desative o Wi-Fi para forçar o tráfego de dados móveis do chip Getnet.
4. **Teste de Comunicação e Atualização de Tabelas:**
   - No menu administrativo/suporte da maquininha (`Menu > Suporte / Funções`), execute o **Teste de Comunicação** para validar o handshake com o autorizador da Getnet.
5. **Acionamento do Suporte Técnico:**
   - Caso o erro persista após validar chip e Wi-Fi, acione o suporte técnico Getnet (Capitais: `4002-4000` ou `4003-4000` / Demais localidades: `0800-648-8000`) ou solicite troca remota de perfil de operadora / substituição de terminal.

## Diagnóstico de Erros de Transação Recusada (Transaction Decline Codes ISO-8583)
Quando uma venda é recusada na maquininha, o visor e o comprovante exibem o código de retorno do emissor ou da rede captura:

- **Código 05 (Não Autorizada pelo Emissor / Do Not Honor):** Recusa preventiva emitida pelo banco emissor do cartão do comprador (ex.: bloqueio temporário no app do banco, suspeita antifraude ou compra fora do perfil). **Ação:** Orientar o portador do cartão a contatar seu banco emissor ou liberar o cartão no aplicativo bancário, ou utilizar outro meio de pagamento (outro cartão ou Pix). Não se trata de defeito físico da maquininha Getnet.
- **Código 51 (Saldo ou Limite Insuficiente / Insufficient Funds):** O cartão do comprador não dispõe de limite de crédito ou saldo em conta para o valor da transação. **Ação:** Solicitar outro cartão, dividir o pagamento ou cobrar via QR Code Pix.
- **Código 54 (Cartão Vencido / Expired Card):** A data de validade do cartão expirou. **Ação:** Solicitar outro cartão válido ao cliente.
- **Código 57 (Transação Não Permitida para o Cartão):** O cartão não está habilitado para aquela modalidade específica (ex.: tentativa de passar crédito em cartão puramente débito/voucher ou compra internacional bloqueada). **Ação:** Verificar a modalidade correta (Débito vs Crédito vs Voucher).
- **Código 12 (Transação Inválida):** Modalidade incompatível ou falha de leitura em pagamento por aproximação (NFC) acima do limite sem senha. **Ação:** Inserir o cartão físico no leitor de chip e digitar a senha.
- **Código 91 / 96 (Emissor Fora do Ar / Falha Temporária de Comunicação):** Instabilidade momentânea no sistema do banco emissor do cartão ou na bandeira. **Ação:** Aguardar alguns instantes e repetir a tentativa ou utilizar outra bandeira/Pix.
- **Venda Duplicada em menos de 5 minutos (Regra Get Smart):** Conforme o FAQ oficial (`site.getnet.com.br/duvidas/`), se ocorrerem duas vendas idênticas no mesmo estabelecimento em intervalo inferior a 5 minutos (mesmo cartão, bandeira e valor), a segunda transação é cancelada automaticamente por proteção contra duplicidade.
