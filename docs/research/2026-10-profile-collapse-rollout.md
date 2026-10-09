# Collapse de perfis: lista kanban-only + dry-run (card 11)

Nada foi aplicado. A decisão de aplicar é do Rodrigo.

## Evidência
Fonte: `sessions.source` em `~/.hermes/profiles/<p>/state.db` (somente leitura), coletado em 2026-10-09.

Kanban-only (nenhuma sessão cli/webui):

| perfil | sessões kanban | última |
|---|---|---|
| monorepo-tooling | 30 | 2026-10-09 02:16 |
| planner | 5 | 2026-09-23 |
| rn-app-engineer | 2 | 2026-09-28 |
| scribe | 2 | 2026-09-28 |

Sem evidência de nenhum uso (state.db sem tabela `sessions`): firmware-engineer, protocol-engineer. Candidatos prováveis, mas não provados kanban-only.

Usados também em chat/CLI (NÃO colapsar):

| perfil | cli | kanban | outros |
|---|---|---|---|
| coder | 490 | 415 | |
| tester | 33 | 92 | |
| trama-engineer | 23 | 418 | webui 1 |
| researcher | 4 | 31 | |
| reviewer | 11 | 45 | |
| auditor | 2 | 5 | |
| delegator | 2 | 1 | |
| obdive-architect | 1 | 1 | |
| ts-core-engineer | 1 | 21 | |

`default` não tem state.db em profiles/ (é o root).

## Dry-run
`python scripts/collapse_profile_routing.py --hermes-home ~/.hermes --dry-run`

O script, como está, marcaria TODOS os 15 perfis (model, fallback_providers, auxiliary.vision.model/provider), inclusive os 9 usados em chat/CLI acima. Isso contradiz a regra do §4 ("colapsar só kanban-only"). O script não tem filtro de perfis.
Nota: os rewrites perdem comentários YAML (backup em `--stamp`).

## Recomendação
1. Antes de aplicar, adicionar ao script um `--only <perfis>` (ou allowlist) e usar: monorepo-tooling, planner, rn-app-engineer, scribe (+ firmware-engineer, protocol-engineer se Rodrigo confirmar).
2. Rodar o dry-run filtrado e só então `--apply --stamp <stamp>`.
