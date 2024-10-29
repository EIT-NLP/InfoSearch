from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class SourceInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name = "Source-v1",
        dataset = {
            "path": "jianqunZ/Source-v1",
            "revision": "67dad8949d168785706d82f93e6794e88c8fcd2e",
        },
        description ="Measuring retrieval instruction following ability on Source-v1.",
        reference = None,
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
        )

