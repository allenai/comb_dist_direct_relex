"""

Unit tests for relation_instances_reader.py

"""

import unittest
from relex.relation_instances_reader import RelationInstancesReader
from allennlp.data.tokenizers.token import Token
class TestUtil(unittest.TestCase):

    def test_entity_distances(self):

        max_dist = 30

        RelationInstancesReader.max_distance = max_dist
        reader = RelationInstancesReader(False, 1, 0, 0, 100, positive_positions=False, cap_positions=True,
                                         binary_classification=False, elmo_indexer=None, supervised='none')
        tokens = [Token(text=x) for x in ['<e1>w0</e1>', 'w1', '<e2>w2</e2>', 'w3', '<e1>w4', 'w5</e1>']]
        expected = ([Token(text=x) for x in ['w0', 'w1', 'w2', 'w3', 'w4', 'w5']],
                    [0, 1, 2, -1, 0, 0], [-2 ,-1, 0, 1, 2, 3],
                    [0, 0, 0, -100, -100, -100], [-100, -100, 0, 0, 0, 0], [-100, -100, -100, -100, 0, 0])
        actual = reader._tokens_distances(tokens)
        assert len(expected) == len(actual)
        assert len(expected) == 6
        assert actual[2] == [x + max_dist for x in expected[2]]
        assert actual[1] == [x + max_dist for x in expected[1]]
        assert actual[3] == expected[3]
        assert actual[4] == expected[4]
        assert actual[5] == expected[5]
        assert [x.text for x in expected[0]] == [x.text for x in actual[0]]

        RelationInstancesReader.max_distance = max_dist
        reader = RelationInstancesReader(False, 1, 0, 0, 100, positive_positions=True, cap_positions=True,
                                         binary_classification=False, elmo_indexer=None, supervised='none')
        tokens = [Token(text=x) for x in ['<e1>w0</e1>', 'w1', '<e2>w2</e2>', 'w3', '<e1>w4', 'w5</e1>']]
        expected = ([Token(text=x) for x in ['w0', 'w1', 'w2', 'w3', 'w4', 'w5']],
                    [0, 1, 2, 1, 0, 0], [2 ,1, 0, 1, 2, 3],
                    [0, 0, 0, -100, -100, -100], [-100, -100, 0, 0, 0, 0], [-100, -100, -100, -100, 0, 0])
        actual = reader._tokens_distances(tokens)
        assert len(expected) == len(actual)
        assert len(expected) == 6
        assert actual[2] == expected[2]
        assert actual[1] == expected[1]
        assert actual[3] == expected[3]
        assert actual[4] == expected[4]
        assert actual[5] == expected[5]
        assert [x.text for x in expected[0]] == [x.text for x in actual[0]]
       
    def test_positions(self):

        RelationInstancesReader.max_distance = 30
        reader = RelationInstancesReader(False, 1, 0, 0, 100, positive_positions=False, cap_positions=True,
                                         binary_classification=False, elmo_indexer=None, supervised='none')
        count = 6
        loc = [(3, 'start'), (3, 'end')]
        expected = [-3, -2, -1, 0, 1, 2]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(3, 'start'), (4, 'end')]
        expected = [-3, -2, -1, 0, 0, 1]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(3, 'start'), (5, 'end')]
        expected = [-3, -2, -1, 0, 0, 0]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(0, 'start'), (0, 'end')]
        expected = [0, 1, 2, 3, 4, 5]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(0, 'start'), (5, 'end')]
        expected = [0, 0, 0, 0, 0, 0]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = []
        expected = [30, 30, 30, 30, 30, 30]
        try:
            actual = reader._positions(count, loc)
            assert False  # missing entity tags, it should throw exception
        except ValueError:
            pass

        count = 6
        loc = [(0, 'start'), (1, 'end'), (5, 'start'), (5, 'end')]
        expected = [0, 0, 1, 2, -1, 0]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(1, 'start'), (1, 'end'), (4, 'start'), (4, 'end')]
        expected = [-1, 0, 1, 1, 0, 1]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 80
        loc = [(40, 'start'), (41, 'end')]
        expected = [-30] * 10 + [x for x in range(-30, 0)] + [0, 0] + [x for x in range(1, 30)] + [30] * 9
        actual = reader._positions(count, loc)
        assert expected == actual

        # Malformed input

        count = 6
        loc = [(3, 'start')]
        expected = [-3, -2, -1, 0, 30, 30]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(3, 'end')]
        expected = [-30, -30, -30, -30, -1, 2]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(2, 'end'), (4, 'start')]
        expected = [-30, -30, -30, -1, 0, 30]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(2, 'start'), (4, 'start')]
        expected = [-2, -1, 0, 30, 30, 30]
        actual = reader._positions(count, loc)
        assert expected == actual

        count = 6
        loc = [(2, 'end'), (4, 'end')]
        expected = [-30, -30, -30, -30, -30, -1]
        actual = reader._positions(count, loc)
        assert expected == actual

