make prog_combo;
mkdir bsub_results;
mkdir bsub_results/combo;
PREFIX="bsub_results/combo";
bsub < combo/lsf_tests/huge_4x1.lsf;
bsub < combo/lsf_tests/huge_4x2.lsf;
bsub < combo/lsf_tests/huge_4x4.lsf;
bsub < combo/lsf_tests/huge_4x8.lsf;
