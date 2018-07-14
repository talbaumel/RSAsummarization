import pyrouge

def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155(rouge_args='-a -c 95 -n 2 -2 4 -u -p 0.5 -l 250')
    #r = pyrouge.Rouge155()

    r.model_filename_pattern = '#ID#_reference_(\d+).txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    return r.convert_and_evaluate()


print rouge_eval('eval/ref/', 'eval/dec/')

#/gttp/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -a -c 95 -n 2 -2 4 -u -p 0.5 -l 250 -m /tmp/tmplGaaxS/rouge_conf.xml


