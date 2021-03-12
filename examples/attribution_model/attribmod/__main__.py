import click

from attribmod.data import create_data
from attribmod.data.data_generation import orchestrate_data_generation
from attribmod.features.sessions_to_cjs import orchestrate_data_processing
from attribmod.models import attribution_fsc, convmod_classifier
from attribmod.models.model_application import orchestrate_model_application
from attribmod.output.format_attribution import orchestrate_output_handling
from attribmod.visualization import am_comparison
from attribmod.utils.env import check_env


@click.group()
def cli():
    pass


@cli.group(help='Create various datasets')
def create():
    pass


@create.command(name='base', help='Create base session data')
def create_base():
    create_data.create_base()


@create.command(name='abt', help='Create analytical base table')
def create_abt():
    create_data.create_abt()


@cli.group(help='Fit various models')
def fit():
    pass


@fit.command(name='convmod', help='Fit the conversion model')
@click.option('-m', '--mlflow-message', default='Aufruf via Kommandozeile')
def fit_convmod(mlflow_message):
    convmod_classifier.fit_classifier(mlflow_comment=mlflow_message)


@fit.command(name='fsc', help='Fit FSC weights')
def fit_fsc():
    attribution_fsc.fit_fsc()


@cli.group(help='Orchestrate pipeline steps')
def orchestrate():
    pass


@orchestrate.command(name='all', help='Execute all steps without training')
def execute_pipeline():
    orchestrate_data_generation()
    orchestrate_data_processing()
    orchestrate_model_application()
    orchestrate_output_handling()


@orchestrate.command(name='extract', help='1st Step: Data generation')
@click.option('--from-date', type=str)
@click.option('--to-date', type=str)
@click.option('--train', is_flag=True)
def data_generation(from_date, to_date, train):
    orchestrate_data_generation(from_date=from_date, to_date=to_date, train=train)


@orchestrate.command(name='transform', help='2nd Step: Process data')
@click.option('--train', is_flag=True)
def data_processing(train):
    orchestrate_data_processing(train=train)


@orchestrate.command(name='model_application', help='3rd Step: Apply model to processed data')
def model_application():
    orchestrate_model_application()


@orchestrate.command(
    name='output_handling', help='4th Step: Format attribution data and write output to table'
)
def output_handling():
    orchestrate_output_handling()


@cli.group(help='Starts dashboard')
def dashboard():
    pass


@dashboard.command(name='compare')
def dashboard_compare():
    am_comparison.start()


if __name__ == '__main__':
    check_env()
    cli()
