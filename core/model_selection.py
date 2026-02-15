import itertools
import numpy as np

def filter_step0_records(records):
    """Filter step 0"""
    return records.filter(lambda r: r['step'] != 0)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionWithEarlyStoppingMethod(SelectionMethod):
    """Picks argmax(val_accuracy), with early stopping"""
    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')


class ProxyAccSelectionMethodWithEarlyStoppingMethod(SelectionMethod):
    """Picks argmax(proxy_acc)"""
    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class ProxyAucSelectionMethodWithEarlyStoppingMethod(SelectionMethod):
    """Picks argmax(proxy_auc)"""
    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')



class ProxyAuc_TestACC(ProxyAucSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAuc_TestACC"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_auc'],'test_acc': record['test_acc']}

class ProxyAuc_TestAUC(ProxyAucSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAuc_TestACC"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_auc'],'test_acc': record['test_auc']}

class ProxyAuc_TestF1(ProxyAucSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAuc_TestF1"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_auc'],'test_acc': record['test_f1']}

class ProxyAuc_TestPrecisi(ProxyAucSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAuc_TestPrecisi"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_auc'],'test_acc': record['test_precision']}

class ProxyAuc_TestRecall(ProxyAucSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAuc_TestRecall"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_auc'],'test_acc': record['test_recall']}



class ProxyAcc_TestACC(ProxyAccSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAcc_TestACC"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_acc'],'test_acc': record['test_acc']}

class ProxyAcc_TestAUC(ProxyAccSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAcc_TestAUC"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_acc'],'test_acc': record['test_auc']}

class ProxyAcc_TestF1(ProxyAccSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAcc_TestF1"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_acc'],'test_acc': record['test_f1']}

class ProxyAcc_TestPrecisi(ProxyAccSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAcc_TestPrecisi"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_acc'],'test_acc': record['test_precision']}

class ProxyAcc_TestRecall(ProxyAccSelectionMethodWithEarlyStoppingMethod):
    name = "ProxyAcc_TestRecall"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_p_acc'],'test_acc': record['test_recall']}



class OracleAcc_TestACC(OracleSelectionWithEarlyStoppingMethod):
    name = "OracleAcc_TestACC"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_o_acc'], 'test_acc': record['test_acc']}

class OracleAcc_TestAUC(OracleSelectionWithEarlyStoppingMethod):
    name = "OracleAcc_TestAUC"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_o_acc'], 'test_acc': record['test_auc']}

class OracleAcc_TestF1(OracleSelectionWithEarlyStoppingMethod):
    name = "OracleAcc_TestF1"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_o_acc'], 'test_acc': record['test_f1']}

class OracleAcc_TestPrecisi(OracleSelectionWithEarlyStoppingMethod):
    name = "OracleAcc_TestPrecisi"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_o_acc'], 'test_acc': record['test_precision']}

class OracleAcc_TestRecall(OracleSelectionWithEarlyStoppingMethod):
    name = "OracleAcc_TestRecall"
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        return {'val_acc':  record['val_o_acc'], 'test_acc': record['test_recall']}

    



