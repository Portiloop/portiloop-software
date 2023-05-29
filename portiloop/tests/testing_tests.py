from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def click_element_by_xpath(driver, xpath, index):
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, xpath)))
    for elem in elements:
        print(elem.text)
    if len(elements) <= index:
        return False
    elements[index].click()
    return True

def click_element_by_name(driver, class_name, text):
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, class_name)))
    for elem in elements:
        if elem.text == text:
            elem.click()
            return True 
    return False 

def click_element_by_index(driver, class_name, index):
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, class_name)))
    if len(elements) <= index:
        return False
    elements[index].click()
    return True

def run_notebook_cell(notebook_url, cell_index):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(notebook_url)

    # Open the Jupyter Notebook in the browser
    driver.get(notebook_url)

    # Wait for the notebook to load
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'notebook-container')))

    # Find the cell by index and execute it
    # cell_css_selector 
    if not click_element_by_index(driver, 'inner_cell', cell_index):
        raise Exception("Issue clicking on cell, index out of range")

    # print(cell_elements)
    actions = ActionChains(driver)
    actions.key_down(Keys.CONTROL).send_keys(Keys.ENTER).key_up(Keys.CONTROL).perform()

    # Wait for the cell execution to complete
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'jupyter-button')))
    time.sleep(1)

    if not click_element_by_name(driver, 'jupyter-widget-Accordion-child', 'Channels'):
        raise Exception("Issue clicking on Channels button")

    time.sleep(1)

    if not click_element_by_xpath(driver, "//input[@value='1']", 0):
        raise Exception("Issue clicking on simple button")

    time.sleep(1)

    # Retrieve the output
    output = driver.find_elements(By.CLASS_NAME, 'output')[cell_index].text

    # Close the web driver
    driver.close()

    return output

cell_index = 0

url = "http://192.168.4.1:8080/notebooks/portiloop-software/portiloop/notebooks/tests.ipynb"

output = run_notebook_cell(url, cell_index)
print("Cell Output:")
print(output)


