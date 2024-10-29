from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class Test2InstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name = "Test-2",
        dataset = {
            "path": "jianqunZ/Test-2",
            "revision": "6f4642a0b4eeaffa89f3350413a5fac487b507a1",
        },
        description ="Measuring retrieval instruction following ability on Test 2.",
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

