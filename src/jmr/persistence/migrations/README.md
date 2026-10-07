# JMR business migrations

The current business schema version is `4`. Version `1` is bootstrapped from
`jmr.persistence.schema.SCHEMA_SQL`; subsequent forward and reversible changes
are declared as numbered migrations in the same module. `apply_schema()` runs
them in order and records each applied version in `jmr.schema_migrations`.

The LangGraph checkpointer and PostgresStore migrations remain owned by their
official packages and are applied through their respective `setup()` methods.
Operational commands and rollback requirements are documented in
`docs/operations.md`.

Future changes must add a monotonically increasing migration instead of
rewriting an already released version. Every forward migration must state
whether it is reversible; irreversible changes require a verified full backup
and restore procedure.
