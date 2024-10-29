from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class LanguageInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name = "Language-v1",
        dataset = {
            "path": "jianqunZ/Language-v1",
            "revision": "d4915806662132e118e8df242aee059ef5669578",
        },
        description ="Measuring retrieval instruction following ability on Language-v1.",
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
