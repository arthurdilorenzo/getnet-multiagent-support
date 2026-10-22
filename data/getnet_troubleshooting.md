# Suporte Técnico e Diagnóstico de Maquininhas Getnet

## Problemas de Conexão com a Internet (Chip Móvel ou Wi-Fi)
Se a maquininha não estiver conectando à internet ou exibir mensagens de "Erro de Conexão / Sem Sinal", siga o protocolo padrão de suporte Getnet:

1. **Verificar os indicadores de sinal no visor:**
   - Conferir se o ícone de Wi-Fi ou GPRS/3G/4G está visível no topo da tela.
   - Observar se há barras de intensidade de sinal. Se estiver com um "X" vermelho ou sem barras, a máquina está sem cobertura.
2. **Reinício de Ciclo de Energia (Soft Reset):**
   - Segure o botão vermelho (Cancelar/Desligar) por 5 segundos até a maquininha desligar por completo.
   - Aguarde 10 segundos e pressione o botão verde para ligar novamente. Isso força o modem a buscar uma nova antena celular da operadora.
3. **Alternância de Conectividade:**
   - Acesse: Menu > Configurações > Conexão / Rede.
   - Se estiver usando o Chip da Operadora, tente conectar em uma rede Wi-Fi local de boa qualidade.
   - Se estiver no Wi-Fi, desative o Wi-Fi para que ela conecte pelo chip multi-operadora da Getnet.
4. **Teste de Comunicação com a Adquirente:**
   - Acesse o Menu > Funções / Utilitários > "Teste de Comunicação".
   - A máquina enviará um pacote de dados para os servidores da Getnet para validar latência e aprovação.
5. **Verificação física do SIM Card:**
   - Com a máquina desligada, abra a tampa traseira, retire o chip com cuidado, limpe os contatos dourados com pano seco e insira novamente com firmeza no slot SIM 1.

## Erros de Transação Recusada (Transaction Decline)
Quando uma transação de venda é recusada na maquininha, a tela exibe um código numérico e mensagem explicativa:
- **Código 51 (Saldo Insuficiente / Not Sufficient Funds):** O cartão do comprador não possui limite ou saldo disponível suficiente para cobrir o valor da compra. Orientar o cliente a utilizar outro cartão ou pagar via Pix.
- **Código 05 (Não Autorizada / Do Not Honor):** Bloqueio de segurança emitido pelo banco emissor do cartão do cliente (por suspeita de fraude, bloqueio temporário no app do banco ou compra fora do padrão). O titular deve entrar no app do próprio banco e desbloquear a transação.
- **Código 12 (Transação Inválida):** Ocorre com frequência ao tentar aproximação (contactless) acima do limite permitido ou cartão sem chip ativo. Solicitar que o cliente insira o cartão físico com chip e digite a senha.
- **Código 91 / 96 (Emissor Indisponível / Erro Temporário):** O sistema do banco do cliente ou da bandeira (Visa, Mastercard, Elo) está momentaneamente fora do ar. Aguardar alguns minutos ou tentar outra bandeira.
