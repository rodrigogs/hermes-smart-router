# Hermes integration research (t_75c80d19)

RELATÓRIO (condensado). Não consegui gravar `docs/research/2026-10-hermes-integration.md`: o perfil researcher é read-only (write_file/patch recusados) e sem terminal. Este comentário é a versão resumida; o relatório completo, com tabela de fontes, está pronto para ser regravado por um perfil com escrita.

## 1. Pontos de extensão nativos (hermes-runtime)
- `pre_kanban_dispatch` retorna só `{model, provider}`, vale para aquele dispatch, nunca grava no DB, e `set-model` humano sempre ganha (kanban_db.py:361-439, plugins.py:270-278). O router já usa (SR/__init__.py:1364, registro :2320).
- NÃO USADOS: `on_kanban_worker_exited` (exit_kind, rate_limited), `kanban_task_completed/blocked`, `on_kanban_dispatch_tick` (plugins.py:279-370). Sem eles o router nunca sabe se o card routeado deu certo.
- Chat: nenhum hook troca o modelo; o router já contorna com middleware llm_request (SR/__init__.py:2284-2293). Correto.
- `smart_model_routing` nativo: só chave de config que o wizard grava desligada (setup.py:3592, config.py:2325). Nada a adotar.
- Copilot: Hermes manda `x-initiator: user` na 1ª chamada de cada turno de usuário, que fatura um premium request (conversation_loop.py:3292-3299, copilot_auth.py:731). Cada worker kanban ou filho `delegate_profile` (`hermes -p X chat -q`) é um turno novo, logo custa ≥1 premium request. O router não modela isso.
- DUPLICADO: (a) registry de preço/contexto `router/capabilities.py:403-871` vs `agent/usage_pricing.py:1263,1448` e `agent/model_metadata.py:1284,2988`; (b) blocos de capacidade copiados ~10x em router.yaml (ex.: gpt-5.4 copilot, router.yaml:20-27); (c) `tiers.*.fallback` vs `fallback_providers` por perfil (coder/config.yaml:4); (d) classificação de quota/breaker vs `agent/error_classifier.py` (extensão não verificada linha a linha).

## 2. Copilot no tier map
Cache real (provider_models_cache.json, at=1791510308), conferido por regex por provedor: copilot lista claude-opus-5.5, claude-sonnet-5.5, claude-haiku-5.5, gpt-6.1-sol, gpt-5.6-sol/terra/luna, gpt-5.5, gpt-5.4, gpt-5.4-mini, gemini-3.8-flash, grok-4.7, kimi-k3, mai-code-1.1-flash. zai tem glm-5.3 e glm-5.3-flash. deepseek tem v4-pro e v4-flash. openai-codex tem gpt-6.1-sol e gpt-5.6-sol/terra/luna.

ACHADO A CONFIRMAR: `gpt-5.5` NÃO aparece na lista de `openai-codex` (0 matches), mas é o primário do perfil coder (profiles/coder/config.yaml:2-3). Ou o cache está defasado ou o coder aponta para modelo que o provedor não lista.

Candidato (só o que muda; todo par model/provider conferido no cache):
- classifier: claude-haiku-5.5/copilot; chain: gpt-5.4-mini/copilot, gpt-5.6-luna/openai-codex, glm-5.3-flash/zai, deepseek-v4-flash. Alternativa se premium requests forem escassos: gpt-5.6-luna/openai-codex.
- fail_safe: gpt-5.6-luna/openai-codex; fallback glm-5.3-flash/zai, claude-haiku-5.5/copilot, deepseek-v4-flash.
- T1: glm-5.3-flash/zai (volume alto, fora do Copilot); fb gpt-5.6-luna, claude-haiku-5.5/copilot, deepseek-v4-flash.
- T2: glm-5.3/zai; fb gpt-5.6-terra, claude-sonnet-5.5/copilot, deepseek-v4-pro.
- T3: claude-sonnet-5.5/copilot (pouco volume, maior ganho); fb gpt-5.6-terra, glm-5.3, deepseek-v4-pro.
- T4: gpt-6.1-sol/openai-codex (já usado por auditor/reviewer); fb claude-opus-5.5/copilot, glm-5.3, deepseek-v4-pro.
- Hop antigravity-local fica fora (não validável no cache, vision:false, max_output 8192).
Pendências: registry não tem linhas para claude-opus-5.5/sonnet-5.5/haiku-5.5, gpt-6.1-sol, gpt-5.4 (capabilities.py:833-871 só tem claude-opus-5, opus-4-8, sonnet-5, haiku-4-5); rodar `router lint` antes. Multiplicadores e cota mensal do Copilot NÃO verificados.

## 3. Observabilidade
Falta: (1) juntar decisão a desfecho via kanban_task_completed/blocked e on_kanban_worker_exited por (task_id, run_id) → taxa de sucesso/bloqueio/rate_limited por tier; (2) campo `source` kanban|chat|delegate no trace; (3) custo/latência por tier lendo usage_pricing (subscription_included em balde próprio) e runs do kanban; (4) alerta de hook inerte no GATEWAY (início do processo vs mtime mais novo do plugin, mais decisão-heartbeat no 1º dispatch_tick pós-boot; o sidecar já expõe process_started_at/code_mtime, mas só vigia a si); (5) 120 de 328 linhas de routes.jsonl citam mimo; filtrar na leitura por "modelo ainda na policy", não apagar (rotação é por tamanho 5MiB x4); (6) lint que confere todo modelo do router.yaml contra o cache. Não medido: rótulo de modo live no trace (regex "mode":"live" deu 0).

## 4. Perfis
Router só muda (model, provider) em kanban e só sem override humano, então o modelo do perfil é piso, não conflito. `scripts/collapse_profile_routing.py` já existe (dry-run) mas parece não aplicado. Não colapsar tudo: perfis usados em chat/CLI ficam sem modelo. Colapsar só perfis kanban-only. Anti-drift por verificação: lint/cron que confere model.default e fallback_providers de cada perfil contra router.yaml e contra o cache.

## 5. Ranking impacto/esforço
1 classificador fora da zai (alto/baixo); 2 conferir gpt-5.5 do coder (alto/baixo); 3 lint de existência de modelo (alto/baixo); 4 desfecho por card (alto/médio); 5 alerta de hook inerte no gateway (médio-alto/baixo-médio); 6 linhas de registry + multiplicadores Copilot (médio/médio); 7 derivar contexto/visão de model_metadata (médio/médio); 8 campo source + filtro de obsoletos (médio/baixo); 9 custo via usage_pricing (médio/alto); 10 modelar x-initiator (médio/médio); 11 cron de drift de perfis e rollout do collapse (médio/médio). Fazer 6 antes de 1.

## 6. Cards sugeridos (não criados), título → aceite
1 Mover classificador p/ provedor ≠ primários → lint limpo, teste afirma provider fora do conjunto de primários, decisão live mostra a chamada.
2 Confirmar modelo do coder (gpt-5.5/openai-codex) → lista viva do provedor comparada ao cache; config e cache concordam.
3 Lint de existência de modelo em router.yaml e perfis vs cache → fixture com modelo inexistente falha; arquivos reais passam; saída nomeia arquivo/chave/modelo.
4 Linhas de registry + multiplicadores Copilot verificados → cada linha cita página oficial e data; cobertura mantida.
5 Gravar desfecho por (task_id, run_id) → endpoint com taxa de sucesso/bloqueio/rate_limited por tier; mutação que não grava o desfecho derruba o teste.
6 Alerta de hook inerte no gateway → dispara após deploy sem restart, silencia após restart; heartbeat de boot registrado.
7 Campo source + filtro de obsoletos → console filtra por source; 120 entradas mimo fora das métricas atuais, legíveis no histórico.
8 Derivar capacidades de model_metadata → teste prova igualdade por modelo; override manual sinalizado.
9 Custo/latência por tier via usage_pricing → console mostra N dias, subscription_included em balde próprio.
10 Cron de drift perfil↔router → relatório agendado; card só se houver mudança.
11 Decidir rollout do collapse → lista de perfis kanban-only; dry-run primeiro.
12 Orçamento de premium requests do Copilot → cota e multiplicadores registrados; shadow de 1 semana mede premium requests por tier; mapa adotado por números.

## Limites
Sem shell: nada veio de saída de comando, só buscas em arquivo e web. Cota/multiplicadores do Copilot, proporção live do trace e sobreposição do classificador de erros não verificados. Docs oficiais lidos só pelos snippets da página de kanban; hooks e contratos vêm do código do runtime.
