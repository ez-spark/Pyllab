delocate-listdeps --all dist/*.whl
delocate-wheel -v -w dist_fixed dist/*.whl
