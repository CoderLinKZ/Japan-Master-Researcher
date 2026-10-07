# File: test_openalex.py
# Author: L1nzhk0
# Purpose: 本文件用于验证 OpenAlex 作者消歧、论文映射和错误降级逻辑。

from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC_ROOT))

from jmr.domain import (  # noqa: E402
    EvidenceVerificationStatus,
    RetrievalStatus,
)
from mcp_servers.scholar import (  # noqa: E402
    OpenAlexPublicationAdapter,
    OpenAlexRequestError,
    PublicationSearchQuery,
)


class FakeJSONHTTPClient:
    """按顺序返回固定 JSON 响应并记录请求。"""

    # 初始化固定响应队列
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str | int]]] = []

    # 记录请求并返回下一条固定 JSON 响应
    def get_json(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> Mapping[str, Any]:
        self.calls.append((url, dict(params)))
        return self._responses.pop(0)


class FailingJSONHTTPClient:
    """模拟 OpenAlex 网络请求失败。"""

    # 为每次请求抛出统一的 OpenAlex 请求错误
    def get_json(
        self,
        url: str,
        params: Mapping[str, str | int],
    ) -> Mapping[str, Any]:
        raise OpenAlexRequestError("OpenAlex test failure")


# 构造 OpenAlex Adapter 测试使用的规范化查询
def search_query() -> PublicationSearchQuery:
    return PublicationSearchQuery.from_arguments(
        {
            "university_name": "Example University",
            "graduate_school_name": "Graduate School of Engineering",
            "professor_name": "山田 太郎",
            "professor_name_variants": ["Taro Yamada"],
            "search_start_date": "2025-01-01",
            "search_end_date": "2026-08-30",
            "max_results": 7,
        }
    )


class OpenAlexPublicationAdapterTests(unittest.TestCase):
    # 验证机构匹配的作者能够用于检索并映射论文候选
    def test_resolves_author_and_maps_works(self) -> None:
        client = FakeJSONHTTPClient(
            [
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A123",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "last_known_institutions": [
                                {
                                    "display_name": "Example University",
                                }
                            ],
                            "affiliations": [],
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W456",
                            "display_name": "Verified Paper",
                            "publication_date": "2026-03-15",
                            "doi": "https://doi.org/10.1000/example",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "Taro Yamada",
                                    },
                                    "institutions": [
                                        {
                                            "display_name": "Example University",
                                        }
                                    ],
                                }
                            ],
                            "primary_location": {
                                "landing_page_url": "https://publisher.test/paper",
                                "source": {"display_name": "Example Journal"},
                            },
                            "abstract_inverted_index": {
                                "Hello": [0],
                                "world": [1],
                            },
                        }
                    ]
                },
            ]
        )
        adapter = OpenAlexPublicationAdapter(client, api_key="test-key")

        result = adapter.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(result.candidates[0].title, "Verified Paper")
        self.assertEqual(
            result.candidates[0].verification_status,
            EvidenceVerificationStatus.VERIFIED,
        )
        self.assertEqual(
            result.candidates[0].identifiers["doi"],
            "10.1000/example",
        )
        self.assertEqual(result.candidates[0].abstract_or_summary, "Hello world")
        self.assertEqual(client.calls[1][1]["per_page"], 7)
        self.assertIn(
            "authorships.author.id:A123",
            str(client.calls[1][1]["filter"]),
        )
        self.assertEqual(client.calls[0][1]["api_key"], "test-key")

    # 验证多个同等姓名候选不会被 Adapter 擅自选中
    def test_returns_confirmation_status_for_ambiguous_authors(self) -> None:
        client = FakeJSONHTTPClient(
            [
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A1",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "last_known_institutions": [],
                        },
                        {
                            "id": "https://openalex.org/A2",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "last_known_institutions": [],
                        },
                    ]
                },
                {"results": []},
                {"results": []},
            ]
        )
        adapter = OpenAlexPublicationAdapter(client)

        result = adapter.search(search_query())

        self.assertEqual(
            result.status,
            RetrievalStatus.NEEDS_USER_CONFIRMATION,
        )
        self.assertEqual(len(client.calls), 3)

    # 验证同机构且共享 ORCID 的 OpenAlex 碎片档案会合并后直接检索
    def test_merges_fragmented_profiles_with_shared_orcid(self) -> None:
        client = FakeJSONHTTPClient(
            [
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A1",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "orcid": "https://orcid.org/0000-0001-2345-6789",
                            "works_count": 120,
                            "last_known_institutions": [
                                {
                                    "display_name": "Example University",
                                }
                            ],
                            "affiliations": [],
                        },
                        {
                            "id": "https://openalex.org/A2",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "orcid": "https://orcid.org/0000-0001-2345-6789",
                            "works_count": 1,
                            "last_known_institutions": [
                                {
                                    "display_name": "Example University",
                                }
                            ],
                            "affiliations": [],
                        },
                        {
                            "id": "https://openalex.org/A3",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "orcid": None,
                            "works_count": 2,
                            "last_known_institutions": [
                                {
                                    "display_name": "Example University",
                                }
                            ],
                            "affiliations": [],
                        },
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "display_name": "Merged Profile Paper",
                            "publication_date": "2026-01-01",
                            "doi": None,
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A1",
                                        "display_name": "Taro Yamada",
                                    },
                                    "institutions": [
                                        {
                                            "display_name": "Example University",
                                        }
                                    ],
                                }
                            ],
                            "primary_location": None,
                            "abstract_inverted_index": None,
                        }
                    ]
                },
            ]
        )
        adapter = OpenAlexPublicationAdapter(client)

        result = adapter.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(result.candidates), 1)
        self.assertIn(
            "authorships.author.id:A1|A2",
            str(client.calls[1][1]["filter"]),
        )
        self.assertNotIn("A3", str(client.calls[1][1]["filter"]))
        self.assertIn("碎片档案", result.warnings[0])

    # 验证目标作者在单篇论文中的机构明确冲突时会剔除同名污染记录
    def test_rejects_work_with_conflicting_target_affiliation(self) -> None:
        client = FakeJSONHTTPClient(
            [
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A123",
                            "display_name": "Taro Yamada",
                            "display_name_alternatives": ["山田 太郎"],
                            "orcid": "https://orcid.org/0000-0001-2345-6789",
                            "works_count": 100,
                            "last_known_institutions": [
                                {
                                    "display_name": "Example University",
                                }
                            ],
                            "affiliations": [],
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W999",
                            "display_name": "Unrelated Same-Name Paper",
                            "publication_date": "2026-01-01",
                            "doi": None,
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A123",
                                        "display_name": "Taro Yamada",
                                    },
                                    "institutions": [
                                        {
                                            "display_name": "Different University",
                                        }
                                    ],
                                }
                            ],
                            "primary_location": None,
                            "abstract_inverted_index": None,
                        }
                    ]
                },
            ]
        )
        adapter = OpenAlexPublicationAdapter(client)

        result = adapter.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.NO_RESULT)
        self.assertEqual(result.candidates, [])
        self.assertIn("明确冲突", result.warnings[-1])

    def test_japanese_university_uses_institution_id_and_name_variant(self) -> None:
        client = FakeJSONHTTPClient(
            [
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A1",
                            "display_name": "Tetsuya Sakai",
                            "orcid": "https://orcid.org/0000-0001-1111-1111",
                            "last_known_institutions": [
                                {
                                    "id": "https://openalex.org/I2",
                                    "display_name": "Cancer Center",
                                }
                            ],
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/I1",
                            "display_name": "Waseda University",
                            "display_name_alternatives": ["早稲田大学"],
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A2",
                            "display_name": "Tetsuya Sakai",
                            "orcid": "https://orcid.org/0000-0002-2222-2222",
                            "last_known_institutions": [
                                {
                                    "id": "https://openalex.org/I1",
                                    "display_name": "Waseda University",
                                }
                            ],
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/A2",
                            "orcid": "https://orcid.org/0000-0002-2222-2222",
                        },
                        {
                            "id": "https://openalex.org/A3",
                            "orcid": "https://orcid.org/0000-0002-2222-2222",
                        },
                    ]
                },
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "display_name": "Information Retrieval Evaluation",
                            "publication_date": "2026-01-01",
                            "authorships": [
                                {
                                    "author": {
                                        "id": "https://openalex.org/A2",
                                        "display_name": "Tetsuya Sakai",
                                    },
                                    "institutions": [
                                        {
                                            "id": "https://openalex.org/I1",
                                            "display_name": "Waseda University",
                                        }
                                    ],
                                }
                            ],
                            "primary_location": None,
                        }
                    ]
                },
            ]
        )
        query = PublicationSearchQuery.from_arguments(
            {
                "university_name": "早稲田大学",
                "graduate_school_name": "基幹理工学研究科",
                "professor_name": "酒井哲也",
                "professor_name_variants": ["Tetsuya Sakai"],
                "search_start_date": "2025-01-01",
                "search_end_date": "2026-08-30",
            }
        )

        result = OpenAlexPublicationAdapter(client).search(query)

        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(
            [item.title for item in result.candidates],
            ["Information Retrieval Evaluation"],
        )
        self.assertIn("authorships.author.id:A2|A3", str(client.calls[4][1]["filter"]))
        self.assertNotIn("A1", str(client.calls[4][1]["filter"]))

    def test_unverified_author_cannot_supply_publication_evidence(self) -> None:
        client = FakeJSONHTTPClient(
            [
                {
                    "results": [
                        {"id": "https://openalex.org/A1", "display_name": "Taro Yamada"}
                    ]
                },
                {"results": []},
                {"results": []},
            ]
        )
        result = OpenAlexPublicationAdapter(client).search(search_query())
        self.assertEqual(result.status, RetrievalStatus.NEEDS_USER_CONFIRMATION)
        self.assertEqual(len(client.calls), 3)

    # 验证 OpenAlex 请求错误会转换为结构化失败而不会向外抛出
    def test_converts_request_error_to_failed_result(self) -> None:
        adapter = OpenAlexPublicationAdapter(FailingJSONHTTPClient())

        result = adapter.search(search_query())

        self.assertEqual(result.status, RetrievalStatus.FAILED)
        self.assertEqual(result.errors, ["OpenAlex test failure"])


if __name__ == "__main__":
    unittest.main()
