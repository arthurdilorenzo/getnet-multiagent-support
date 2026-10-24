# AF05 — revisão focada

Revisão de SPEC_AF05_REMAINDER.md e /private/tmp/af05-repair.diff; sem suíte, rede ou execução nativa. Não atribui nota. Recibos 1551/42 reportados pelo integrador não foram reproduzidos; final 1555 ainda pendente neste parecer.

## P2 fechado no recheck — proposta herda contexto anterior inteiro (histórico)

Em CustomerSupportAgent._authorized_remote_action, quando a diretiva de proposta não contém ID e há um único alvo global, `action_context = norm[:norm.find(clause)] + clause` considera TODO o prefixo, embora a spec e o comentário prometam somente a frase anterior. Assim uma ação histórica remota pode autorizar proposta após mudança de assunto: `We considered a reboot of GET-CLASSICA-10442 last year. Now discuss paper rolls. Register a proposal.` O código reconhece remote_reboot e retorna True, apesar de a frase imediatamente anterior não conter ação/alvo e a nova proposta estar ambígua. O exemplo é reprodução lógica da ramificação, não execução observada neste review. Limitar herança ao contexto imediatamente anterior e negar quando não contiver par ação/alvo inequívoco; preservar os negativos existentes. Isso persiste só proposta, não demonstra execução não consentida, mas ainda excede o contrato de consentimento de proposta definido.

## Demais mudanças no recorte

Fanout só ocorre depois de seleção da família diagnose, respeita targets atribuídos, constraints, orçamento e resultados prévios sem erro; não introduz execução de escrita. Preflight all-foreign é anterior ao FC/registry e conserva proteção do handler; caso misto permanece protegido por alvo no constrain_call. Formato cells não autoriza campos novos nem ações dos negativos adicionados. Proposta mantém confirmação JWT/hash/idempotência fora do agente. Não identifiquei outro bloqueador concreto neste diff restrito, sem certificar linguagem irrestrita ou qualidade Gemini.

## Recheck restrito do P2

A correção foi lida no trecho `_authorized_remote_action`: substitui prefixo global por lista/index de cláusulas, permite apenas a cláusula imediatamente anterior, exige que ela contenha o conjunto do único alvo explícito e inicie com pedido/comando reconhecido. O exemplo histórico com mudança de assunto deixa de fornecer autoridade. O recibo `verification/af05/review_red.txt` preserva o RED exato antes da correção. Não rodei testes; o foco GREEN estava em andamento pelo integrador. P2 fechado por inspeção da correção no escopo reportado, sem outro bloqueador observado neste recorte e sem nota nova.
