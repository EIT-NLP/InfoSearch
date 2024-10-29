from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class FormatInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name = "Format-v1",
        dataset = {
            "path": "jianqunZ/Format-v1",
            "revision": "63b53c9e117fa16a469026650861db77f25ab823",
        },
        description ="Measuring retrieval instruction following ability on Format-v1.",
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
