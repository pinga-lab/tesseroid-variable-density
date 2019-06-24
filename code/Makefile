# Manages building, testing, and cleaning the code as well as running the code
# to generate the results and figures for the paper.

# CONFIGURATION
###############################################################################

# Set the package name
PACKAGE = tesseroid_density

TESTDIR = tests
SCRIPTSDIR = scripts
RESDIR = $(SCRIPTSDIR)/results

SCRIPTS = $(SCRIPTSDIR)/computation_time.py             \
          $(SCRIPTSDIR)/density-based-discretization.py \
          $(SCRIPTSDIR)/exponential_density.py          \
          $(SCRIPTSDIR)/grid_search.py                  \
          $(SCRIPTSDIR)/linear_density.py               \
          $(SCRIPTSDIR)/neuquen_basin.py                \
          $(SCRIPTSDIR)/sine_density.py

RESULTS = $(RESDIR)/linear           \
		  $(RESDIR)/grid-search      \
		  $(RESDIR)/exponential      \
		  $(RESDIR)/sine             \
		  $(RESDIR)/computation_time \
		  $(RESDIR)/neuquen          \

PYTEST_ARGS = --doctest-modules -v
PYTEST_COV_ARGS = --cov-report=term-missing

RUN = python


# TARGETS
###############################################################################

help:
	@echo "Commands:"
	@echo ""
	@echo "  all            runs 'build', 'test', and 'figures'"
	@echo "  build          build and install the package"
	@echo "  test           run the test suite (including doctests)"
	@echo "  coverage       calculate test coverage"
	@echo "  check          run all code quality checks (pep8, linter)"
	@echo "  pep8           check for PEP8 style compliance"
	@echo "  lint           run static analysis using pylint"
	@echo "  clean          clean up the package build"
	@echo "  results        execute all scripts (generating results)"
	@echo "  figures        execute all scripts (not generating results)"
	@echo ""


# Clean does not remove results or figure files to avoid accidentally deleting
# the results of long running computations.
clean:
	find . -name "*.pyc" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache


# Run all steps to generate the final figures. Results files are generated
# based on the figure requirements (see below).
all: build test figures


# Generating figures
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

figures:
	$(foreach script, $(SCRIPTS), $(RUN) $(script);)


# Generating results
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

results:
	$(foreach resultdir, $(RESULTS), rm -r $(resultdir);)
	$(foreach script, $(SCRIPTS), $(RUN) $(script);)


# Building and testing the code
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

build:
	pip install --no-deps -e .

test:
	pytest $(PYTEST_ARGS) $(PACKAGE) $(TESTDIR)

coverage:
	pytest $(PYTEST_COV_ARGS) --cov=$(PACKAGE) $(PYTEST_ARGS) $(PACKAGE) $(TESTDIR)

check: pep8 lint

pep8:
	flake8 $(PACKAGE) $(TESTDIR) setup.py

lint:
	pylint $(PACKAGE) setup.py
