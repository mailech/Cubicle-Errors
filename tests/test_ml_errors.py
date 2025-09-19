
import pytest


def test_dataset_len():
    from ml_errors.dataloader import ToyDataset

    ds = ToyDataset([1, 2, 3])
    assert len(ds) == 3  # will fail (len returns 4)


def test_training_loop():
    import ml_errors.train as tr

    # This should raise due to shape mismatch; assert raises to mark failure explicitly
    with pytest.raises(Exception):
        tr.train_one_epoch()
