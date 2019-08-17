
ORIGINALS = $(wildcard figure/*-1.pdf)
TRIMMED_PLOTS = $(patsubst %-1.pdf,%-1-cropped.pdf,$(ORIGINALS))

#%-1-cropped.pdf: %-1.pdf
#	convert -density 150 -fuzz 28% -trim +repage -quality 100 -flatten -sharpen 0x1.0 $(patsubst %-1-cropped.pdf,%-1.pdf,$@) $@

all: output/ieee-paper.pdf #output/ieee-paper.tex output/lncs-paper.pdf output/release-paper.pdf output/double-blind-release-paper.pdf output/paper.tex

#output/paper.md: paper/paper.Rmd
#	mkdir -p output
#	Rscript -e "library(knitr); knit(input='paper/paper.Rmd',output='output/paper.md')"

output/ieee-paper.pdf output/ieee-paper.tex: paper/paper.md ieee-packages.yaml bibliography/bibliography.bib templates/ieee-longtable-fix-preamble.latex
	cp ./styles/IEEEtran.cls .
	mkdir -p output
	pandoc  --wrap=preserve \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--filter ./pandoc-tools/bib-filter.py \
		--number-sections \
		--csl=./styles/ieee.csl \
		./ieee-packages.yaml \
		--include-before-body=./templates/ieee-longtable-fix-preamble.latex \
		--include-before-body=./ieee-author-preamble.latex \
		--template=./templates/ieee.latex \
		-o output/ieee-paper.$(subst output/ieee-paper.,,$@) paper/paper.md
	rm ./IEEEtran.cls

grammarly: output/paper.md
	pkill Grammarly || true #if grammarly already exists kill it
	pandoc  --wrap=preserve \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--number-sections \
		-t plain \
		-o output/paper.txt output/paper.md #now get just the text
	open -a Grammarly output/paper.txt #and open it in grammarly

output/acm-paper.pdf output/acm-paper.tex: output/paper.md
	cp ./styles/acm.cls .
	pandoc  --wrap=preserve \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--csl=./styles/acm.csl \
		--number-sections \
		./acm-packages.yaml \
		--template=./templates/acm.latex \
		-o output/acm-paper.$(subst output/acm-paper.,,$@) output/paper.md
	rm ./acm.cls

arXiv.tar: output/acm-paper.tex
	cp output/acm-paper.tex .
	cp styles/acm.cls .
	tar -cvf arXiv.tar ./document.tex ./figure/draw_stacked_plots-1.pdf ./figure/draw_lud_diagonal_internal_all_kiviat-1.pdf ./figure/draw_lud_diagonal_perimeter_lmae_all_kiviat-1.pdf ./acm.cls
	rm ./acm-paper.tex ./acm.cls

output/lncs-paper.pdf output/lncs-paper.tex: output/paper.md
	cp ./styles/llncs.cls .
	pandoc  --wrap=preserve \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--csl=./styles/lncs.csl \
		--number-sections \
		./llncs-packages.yaml \
		--template=./templates/llncs.latex \
		-o output/lncs-paper.$(subst output/lncs-paper.,,$@) output/paper.md
	rm ./llncs.cls

#output/release-paper.pdf: output/paper.md
#	cp ./styles/llncs.cls .
#	pandoc  --wrap=preserve \
#		--filter pandoc-crossref \
#		--filter pandoc-citeproc \
#		--csl=./styles/lncs.csl \
#		--number-sections \
#		./release-packages.yaml \
#		--template=./templates/llncs.latex \
#		-o output/release-paper.pdf output/paper.md
#	rm ./llncs.cls
#
#output/double-blind-release-paper.pdf: output/paper.md $(TRIMMED_PLOTS)
#	cp ./styles/llncs.cls .
#	pandoc  --wrap=preserve \
#		--filter pandoc-crossref \
#		--filter pandoc-citeproc \
#		--csl=./styles/lncs.csl \
#		--number-sections \
#		./double-blind-release-packages.yaml \
#		--template=./templates/llncs.latex \
#		-o output/double-blind-release-paper.pdf output/paper.md
#	rm ./llncs.cls

clean:
	rm output/*

