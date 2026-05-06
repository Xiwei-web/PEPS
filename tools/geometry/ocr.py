"""OCR tool slot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.tools.base import BaseTool, LocalModelRequirement, ToolContext, ToolResult, ToolSpec
from peps.tools.geometry.detection import BoundingBox2D


@dataclass(slots=True)
class OCRText:
    text: str
    box: BoundingBox2D | None = None
    score: float = 1.0


@dataclass(slots=True)
class OCRResult:
    texts: list[OCRText] = field(default_factory=list)
    backend: str = "placeholder"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        if not self.texts:
            return "OCRResult: no text found by the current placeholder backend."
        return "OCRResult: " + ", ".join(item.text for item in self.texts[:5])


class OCRTool(BaseTool):
    spec = ToolSpec(
        name="ocr",
        description="Read visible text from an image.",
        input_schema={
            "type": "object",
            "required": ["image_source"],
            "properties": {
                "image_source": {"type": "string"},
                "precomputed_texts": {"type": "array"},
            },
        },
        output_schema={"type": "object", "class": "OCRResult"},
        model_requirements=[
            LocalModelRequirement(
                name="EasyOCR",
                required_for="local optical character recognition",
                local_path_env="PEPS_EASYOCR_MODEL_DIR",
                package="easyocr",
                download_hint="EasyOCR downloads recognition weights on first use unless cached. Configure its cache in README later.",
                required=False,
            )
        ],
        tags=["perception", "ocr"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        image_source: Any,
        precomputed_texts: list[Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        texts: list[OCRText] = []
        for row in precomputed_texts or []:
            if isinstance(row, OCRText):
                texts.append(row)
            elif isinstance(row, dict):
                box = row.get("box")
                bbox = BoundingBox2D(*box, label="text", score=row.get("score", 1.0)) if box else None
                texts.append(OCRText(text=row["text"], box=bbox, score=float(row.get("score", 1.0))))
            else:
                texts.append(OCRText(text=str(row)))
        result = OCRResult(
            texts=texts,
            backend="precomputed_or_placeholder",
            metadata={
                "image_source": str(image_source),
                "note": "Configure EasyOCR backend for real OCR.",
            },
        )
        return self.success(result)
