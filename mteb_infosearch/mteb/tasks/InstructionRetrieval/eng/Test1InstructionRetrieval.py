from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class Test1InstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name = "Test-1",
        dataset = {
            "path": "jianqunZ/Test-1",
            "revision": "f500d43f19ca78e9a77eccd45a72fb0edbb4387b",
        },
        description ="Measuring retrieval instruction following ability on Test 1.",
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

