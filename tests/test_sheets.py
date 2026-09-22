"""Product sheets are read the same whatever markup holds them, and blends stay apart."""

from mlops_core.data.sheets import headed_paragraphs, html_text, labelled_lines, records

LABELS: dict[str, str | None] = {
    "Country": "country",
    "Region": "region",
    "Producer": "producer",
    "Producers": "producer",
    "Roaster": None,  # read so it ends the value before it, then dropped
}


def test_markup_becomes_one_line_per_block() -> None:
    markup = "<ul><li>Country:&nbsp;Kenya</li><li>Region: Nyeri<br>\n</li></ul><p>  a   b </p>"

    assert html_text(markup) == "Country: Kenya\nRegion: Nyeri\na b"
    assert html_text(None) == ""


def test_labels_are_read_in_order_whatever_their_case() -> None:
    text = "A washed lot.\nCOUNTRY: Kenya\nregion : Nyeri, Kirinyaga\nProducers: A and B"

    assert labelled_lines(text, LABELS) == [
        ("country", "Kenya"),
        ("region", "Nyeri, Kirinyaga"),
        ("producer", "A and B"),  # the longer label, not "Producer" and a stray "s:"
    ]


def test_a_value_ends_at_the_next_label_even_on_the_same_line() -> None:
    """Some sheets run their labels together: 'YemenRegion: Haraz' is two values."""
    text = "Country: YemenRegion: HarazRoaster: Somebody's"

    assert labelled_lines(text, LABELS) == [("country", "Yemen"), ("region", "Haraz")]


def test_a_label_with_nothing_after_it_is_skipped() -> None:
    assert labelled_lines("Country:\nRegion: Huye", LABELS) == [("region", "Huye")]


def test_headed_paragraphs_are_read_through_one_wrapper() -> None:
    page = (
        "<h2>You may also like</h2><p>Other coffee</p>"
        "<h5>Region</h5><div class='inside'><p>Huautla, Oaxaca</p><img/></div>"
        "<h5>Producers</h5>\n<p>A <b>union</b></p>"
    )

    assert headed_paragraphs(page, LABELS) == [
        ("region", "Huautla, Oaxaca"),
        ("producer", "A union"),
    ]
    assert headed_paragraphs(None, LABELS) == []


def test_a_field_seen_again_starts_the_next_record() -> None:
    """A blend lists each component with the same labels."""
    pairs = [("region", "Oaxaca"), ("producer", "A"), ("region", "Chiapas"), ("producer", "B")]

    assert records(pairs) == [
        {"region": "Oaxaca", "producer": "A"},
        {"region": "Chiapas", "producer": "B"},
    ]
    assert records([]) == []
