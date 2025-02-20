from src.data.eval import DSLTL, FRMT, NTREX, Publico


class TestPublico:
    def test_init(self):
        dataset = Publico()
        assert len(dataset) == 6_832

    def test_init_train(self):
        dataset = Publico("train")
        assert len(dataset) == 27_325

    def test_iter(self):
        dataset = Publico()
        for id, src, tgt in dataset:
            assert isinstance(id, str)
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            break

    def test_getitem(self):
        dataset = Publico()
        id, src, tgt = dataset[0]
        assert isinstance(id, str)
        assert isinstance(src, str)
        assert isinstance(tgt, str)

    def test_target(self):
        dataset = Publico()
        assert isinstance(dataset.target, list)
        assert isinstance(dataset.target[0], str)

    def test_source(self):
        dataset = Publico()
        assert isinstance(dataset.source, list)
        assert isinstance(dataset.source[0], str)

    def test_batch(self):
        dataset = Publico()
        for idxs, batch in dataset.batch(2):
            assert isinstance(idxs, list)
            assert isinstance(batch, list)
            assert len(idxs) == 2
            assert len(batch) == 2
            break


class TestDSLTL:
    def test_init(self):
        dataset = DSLTL()
        assert len(dataset) == 857

    def test_iter(self):
        dataset = DSLTL()
        for id, src, tgt in dataset:
            assert isinstance(id, int)
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            break

    def test_getitem(self):
        dataset = DSLTL()
        id, src, tgt = dataset[0]
        assert isinstance(id, int)
        assert isinstance(src, str)
        assert isinstance(tgt, str)

    def test_target(self):
        dataset = DSLTL()
        assert isinstance(dataset.target, list)
        assert isinstance(dataset.target[0], str)

    def test_source(self):
        dataset = DSLTL()
        assert isinstance(dataset.source, list)
        assert isinstance(dataset.source[0], str)


class TestFRMT:
    def test_init(self):
        dataset = FRMT()
        assert len(dataset) == 2_609

    def test_iter(self):
        dataset = FRMT()
        for id, src, tgt in dataset:
            assert isinstance(id, int)
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            break

    def test_getitem(self):
        dataset = FRMT()
        id, src, tgt = dataset[0]
        assert isinstance(id, int)
        assert isinstance(src, str)
        assert isinstance(tgt, str)

    def test_target(self):
        dataset = FRMT()
        assert isinstance(dataset.target, list)
        assert isinstance(dataset.target[0], str)

    def test_source(self):
        dataset = FRMT()
        assert isinstance(dataset.source, list)
        assert isinstance(dataset.source[0], str)


class TestNTREX:
    def test_init(self):
        dataset = NTREX()
        assert len(dataset) == 1_997

    def test_iter(self):
        dataset = NTREX()
        for id, src, tgt in dataset:
            assert isinstance(id, int)
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            break

    def test_getitem(self):
        dataset = NTREX()
        id, src, tgt = dataset[0]
        assert isinstance(id, int)
        assert isinstance(src, str)
        assert isinstance(tgt, str)

    def test_target(self):
        dataset = NTREX()
        assert isinstance(dataset.target, list)
        assert isinstance(dataset.target[0], str)

    def test_source(self):
        dataset = NTREX()
        assert isinstance(dataset.source, list)
        assert isinstance(dataset.source[0], str)
