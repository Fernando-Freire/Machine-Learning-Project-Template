################################################################################
##                                   COMMANDS                                 ##
################################################################################

MAKE += --no-print-directory RECURSIVE=1

ifndef VERBOSE
COMPOSE := docker-compose 2>/dev/null
COMPOSE_BUILD := $(COMPOSE) build -q
else
COMPOSE := docker-compose
COMPOSE_BUILD := $(COMPOSE) build
endif

################################################################################
##                                   COLORS                                   ##
################################################################################

RES := \033[0m
MSG := \033[1;36m
ERR := \033[1;31m
SUC := \033[1;32m
WRN := \033[1;33m
NTE := \033[1;34m

################################################################################
##                                 AUXILIARY                                  ##
################################################################################

# Variable do allow is-empty and not-empty to work with ifdef/ifndef
export T := 1

define is-empty
$(strip $(if $(strip $1),,T))
endef

define not-empty
$(strip $(if $(strip $1),T))
endef

define message
printf "${MSG}%s${RES}\n" $(strip $1)
endef

define success
(printf "${SUC}%s${RES}\n" $(strip $1); exit 0)
endef

define warn
(printf "${WRN}%s${RES}\n" $(strip $1); exit 0)
endef

define failure
(printf "${ERR}%s${RES}\n" $(strip $1); exit 1)
endef

define note
(printf "${NTE}%s${RES}\n" $(strip $1); exit 0)
endef

################################################################################
##                                   AWS                                      ##
################################################################################

SHELL := /bin/bash

# Do not execute in recursive calls or within Jenkins
ifdef $(call is-empty,${RECURSIVE})
export AWS_ACCESS_KEY_ID := $(shell aws configure get aws_access_key_id)
export AWS_SECRET_ACCESS_KEY := $(shell aws configure get aws_secret_access_key)
export AWS_SESSION_TOKEN := $(shell aws configure get aws_session_token)
endif

################################################################################
##                                DOCKER BUILD                                ##
################################################################################

build-jupyter:
	@$(call message,"Construindo imagem docker para notebooks")
	@$(COMPOSE_BUILD) jupyter

build-scripts:
	@$(call message,"Construindo imagem docker para teste de scripts")
	@$(COMPOSE_BUILD) scripts

build:
	@$(MAKE) build-jupyter
	@$(MAKE) build-scripts


#bump-and-release:
#	@$(call message,"Fazendo bump ${KIND} da versão do projeto")
#	@(cd  && poetry version ${KIND})
#	@$(MAKE) release-image TAG=`(cd bela_gil && poetry version -s)`
#	@git add pyproject.toml
#	@git commit -m "Release version: v`(cd bela_gil && poetry version -s)`"
#	@git tag v`(cd bela_gil && poetry version -s)`
#	@git push origin HEAD --tags

################################################################################
##                               LINT & FORMAT                                ##
################################################################################

isort:
	@$(call message,"Rodando isort")
	@$(COMPOSE) run -T --rm --entrypoint isort scripts scripts

black:
	@$(call message,"Rodando black")
	@$(COMPOSE) run -T --rm --entrypoint black scripts scripts

autoflake:
	@$(call message,"Rodando autoflake")
	@$(COMPOSE) run --rm --entrypoint autoflake scripts \
		--in-place --remove-all-unused-imports --remove-unused-variables \
		--ignore-init-module-imports --expand-star-imports --recursive \
		scripts

mypy:
	@$(call message,"Rodando mypy")
	@$(COMPOSE) run -T --rm --entrypoint mypy scripts scripts

flake8:
	@$(call message,"Rodando flake8")
	@$(COMPOSE) run -T --rm --entrypoint flake8 scripts scripts

lint:
	@$(MAKE) mypy
	@$(MAKE) flake8

format:
	@$(MAKE) black
	@$(MAKE) isort
	@$(MAKE) autoflake


################################################################################
##                                 TESTS                                      ##
################################################################################

unit-tests:
	@$(COMPOSE) run --rm --entrypoint pytest scripts -v scripts/tests/unit

integration-tests:
	@$(COMPOSE) run --rm --entrypoint pytest scripts -v scripts/tests/integration

check-bow-dtc-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_bow_dtc.py \
		&& $(call success,"Teste passou! Teste de código dos modelos bow+dtc obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos bow+dtc não obteve métricas aceitaveis")

check-tfidf-dtc-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_tfidf_dtc.py \
		&& $(call success,"Teste passou! Teste de código dos modelos tfidf+dtc obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos tfidf+dtc não obteve métricas aceitaveis")

check-bow-etc-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_bow_etc.py \
		&& $(call success,"Teste passou! Teste de código dos modelos bow+etc obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos bow+etc não obteve métricas aceitaveis")

check-bow-lgb-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_bow_lgb.py \
		&& $(call success,"Teste passou! Teste de código dos modelos bow+lgb obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos bow+lgb não obteve métricas aceitaveis")

check-Word2Vec-lgb-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_Word2Vec_lgb.py \
		&& $(call success,"Teste passou! Teste de código dos modelos Word2Vec+lgb obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos Word2Vec+lgb não obteve métricas aceitaveis")

check-Doc2Vec-lgb-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_Doc2Vec_lgb.py \
		&& $(call success,"Teste passou! Teste de código dos modelos Doc2Vec+lgb obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos Doc2Vec+lgb não obteve métricas aceitaveis")

check-FastText-lgb-training:
	@$(COMPOSE) run --rm --entrypoint python scripts scripts/tests/helpers/check_FastText_lgb.py \
		&& $(call success,"Teste passou! Teste de código dos modelos FastText+lgb obteve métricas aceitaveis") \
	  	|| $(call failure,"Teste falhou! Teste de código dos modelos FastText+lgb não obteve métricas aceitaveis")

e2e-tests:
	@$(MAKE) check-bow-dtc-training
	@$(MAKE) check-tfidf-dtc-training
	@$(MAKE) check-bow-etc-training
	@$(MAKE) check-bow-lgb-training
	@$(MAKE) check-Word2Vec-lgb-training
	@$(MAKE) check-FastText-lgb-training
	@$(MAKE) check-Doc2Vec-lgb-training

all-tests:
	@$(MAKE) unit-tests
	@$(MAKE) integration-tests
	@$(MAKE) e2e-tests