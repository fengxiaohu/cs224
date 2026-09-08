PY ?= python3
SYNC = $(PY) scripts/sync.py

.PHONY: sync retry list clean-cache

## Sync: fetch index, parse, download missing files, regenerate markdown
sync:
	$(SYNC)

## Retry downloads that previously failed
retry:
	$(SYNC) --retry-failed

## Parse and print the manifest without downloading
list:
	$(SYNC) --list

## Remove downloaded artifacts (keep metadata + generated markdown)
clean-cache:
	rm -rf lectures assignments handouts
	rm -f _metadata/download.log _metadata/failed.txt
	@echo "cleaned downloads (resources.json/urls.txt kept)"