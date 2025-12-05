make prog_combo;
mkdir bsub_results;
mkdir bsub_results/combo;
PREFIX="bsub_results/combo";
bsub < combo/lsf_tests/small_2x1.lsf;
bsub < combo/lsf_tests/small_2x2.lsf;
bsub < combo/lsf_tests/small_2x4.lsf;
bsub < combo/lsf_tests/small_2x8.lsf;
