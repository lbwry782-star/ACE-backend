"""
Builder1 product modality classification tests.

Run: python -m unittest tests.test_builder1_product_modality -v
"""
from __future__ import annotations

import unittest

from engine.builder1_compliance_product_grounding import (
    AdvertisedProductType,
    classify_advertised_product_type,
)
from engine.builder1_product_modality import (
    ProductModality,
    derive_product_modality,
    resolve_product_modality,
)


class TestServiceDetection(unittest.TestCase):
    def test_amir_gottlieb_hebrew_tutor_is_service(self) -> None:
        self.assertEqual(
            derive_product_modality(
                product_name="אמיר גוטליב",
                product_description="מורה פרטי להיסטוריה. מכין תלמידים לבגרות בהיסטוריה.",
            ),
            ProductModality.SERVICE,
        )

    def test_hebrew_female_private_math_tutor(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="מורה פרטית למתמטיקה"),
            ProductModality.SERVICE,
        )

    def test_hebrew_private_english_lessons(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="שיעורים פרטיים באנגלית"),
            ProductModality.SERVICE,
        )

    def test_english_private_history_tutor(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="Private history tutor"),
            ProductModality.SERVICE,
        )

    def test_teacher_preparing_students_for_exams(self) -> None:
        self.assertEqual(
            derive_product_modality(
                product_description="Teacher preparing students for exams"
            ),
            ProductModality.SERVICE,
        )

    def test_consulting_firm_is_service(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="Consulting firm for strategy"),
            ProductModality.SERVICE,
        )

    def test_online_tutoring_service_not_digital(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="Online tutoring service"),
            ProductModality.SERVICE,
        )


class TestNonServiceModalities(unittest.TestCase):
    def test_physical_book(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="History textbook for students"),
            ProductModality.PHYSICAL_PRODUCT,
        )

    def test_hebrew_study_book(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="ספר לימוד היסטוריה"),
            ProductModality.PHYSICAL_PRODUCT,
        )

    def test_food_drink_physical(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="Organic food and drink"),
            ProductModality.PHYSICAL_PRODUCT,
        )

    def test_software_app_digital(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="Software app for productivity"),
            ProductModality.DIGITAL_PRODUCT,
        )

    def test_hebrew_app_digital(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="אפליקציה ללימוד היסטוריה"),
            ProductModality.DIGITAL_PRODUCT,
        )

    def test_organization_english(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="Acme brand organization"),
            ProductModality.ORGANIZATION,
        )

    def test_hebrew_school_organization(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="בית ספר להיסטוריה"),
            ProductModality.ORGANIZATION,
        )

    def test_shoe_with_incidental_service_word_stays_physical(self) -> None:
        self.assertEqual(
            derive_product_modality(
                product_description="Premium shoe with lifetime repair service"
            ),
            ProductModality.PHYSICAL_PRODUCT,
        )


class TestResolveAndLegacy(unittest.TestCase):
    def test_resolve_uses_stored_modality_on_resume(self) -> None:
        self.assertEqual(
            resolve_product_modality(
                product_name="אמיר גוטליב",
                product_description="מורה פרטי להיסטוריה",
                planning_internals={"productModality": "PHYSICAL_PRODUCT"},
            ),
            ProductModality.PHYSICAL_PRODUCT,
        )

    def test_resolve_derives_service_for_new_assembly(self) -> None:
        self.assertEqual(
            resolve_product_modality(
                product_name="אמיר גוטליב",
                product_description="מורה פרטי להיסטוריה. מכין תלמידים לבגרות בהיסטוריה.",
                planning_internals={},
            ),
            ProductModality.SERVICE,
        )

    def test_named_person_precedence_unchanged(self) -> None:
        self.assertEqual(
            classify_advertised_product_type(
                product_name="אמיר גוטליב",
                product_description="מורה פרטי להיסטוריה. מכין תלמידים לבגרות בהיסטוריה.",
            ),
            AdvertisedProductType.NAMED_PERSON,
        )


class TestPlanningCallCountsUnchanged(unittest.TestCase):
    def test_supplied_name_planning_count_remains_five(self) -> None:
        from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME

        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)

    def test_generated_name_planning_count_remains_six(self) -> None:
        from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME

        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 6)


if __name__ == "__main__":
    unittest.main()
