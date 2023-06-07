from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import random


def click_element_by_xpath(driver, xpath, index):
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, xpath)))
    if len(elements) <= index:
        raise Exception(f"Issue clicking on button by index. xpath: {xpath}, index: {index}, number of elements found: {len(elements)}")
    elements[index].click()
    return True

def click_element_by_name(driver, class_name, text):
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, class_name)))
    for elem in elements:
        if elem.text == text:
            elem.click()
    # raise Exception(f"Issue clicking on button by text. class_name: {class_name}, text: {text}")

def click_element_by_index(driver, class_name, index):
    elements = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, class_name)))
    if len(elements) <= index:
        raise Exception(f"Issue clicking on button by index. class_name: {class_name}, index: {index}, number of elements found: {len(elements)}")
    elements[index].click()


def check_output(output):
    """Check if the output of the function does not contain any errors

    Args:
        output (_type_): string of all output of the test

    Returns:
        bool: indicates if the test passed or not.
    """
    for line in output:
        if "warning" in line.lower() or "error" in line.lower():
            return False
        
    return True

NUM_CHANNELS = 4

# Add all the clickable elements to the clickable dictionary
clickable_elements = {
    "filter": (lambda driver: click_element_by_xpath(driver, "//span[@title='Filter']", 0)),
    "detect": (lambda driver: click_element_by_xpath(driver, "//span[@title='Detect']", 0)),
    "stimulate": (lambda driver: click_element_by_xpath(driver, "//span[@title='Stimulate']", 0)),
    "record_edf": (lambda driver: click_element_by_xpath(driver, "//span[@title='Record EDF']", 0)),
    "stream_lsl": (lambda driver: click_element_by_xpath(driver, "//span[@title='Stream LSL']", 0)),
    "test_stimulus": (lambda driver: click_element_by_xpath(driver, "//button[@title='Send a test stimulus']", 0)),
    "channels_section": (lambda driver: click_element_by_name(driver, 'jupyter-widget-Accordion-child', 'Channels')),
    "start": (lambda driver: click_element_by_xpath(driver, "//button[@title='Start capture']", 0)),
    "stop": (lambda driver: click_element_by_xpath(driver, "//button[@title='Stop capture']", 0)),
    "active": (lambda driver: click_element_by_xpath(driver, "//button[@title='Detector and stimulator active']", 0)),
    "paused": (lambda driver: click_element_by_xpath(driver, "//button[@title='Detector and stimulator paused']", 0)),
    "input_ads": (lambda driver: click_element_by_xpath(driver, "//button[@title='Read data from ADS.']", 0)),
    "input_file": (lambda driver: click_element_by_xpath(driver, "//button[@title='Read data from file.']", 0)),
    "60hz": (lambda driver: click_element_by_xpath(driver, "//button[@title='North America 60 Hz']", 0)), 
    "50hz": (lambda driver: click_element_by_xpath(driver, "//button[@title='Europe 50 Hz']", 0)),
    "clock_coral": (lambda driver: click_element_by_xpath(driver, "//button[@title='Use Coral clock (very precise, not very timely)']", 0)),
    "clock_ads": (lambda driver: click_element_by_xpath(driver, "//button[@title='Use ADS clock (not very precise, very timely)']", 0)),
}

channel_selector = {}
for i in range(NUM_CHANNELS):
    channel_selector[f"ch{i+1}_disabled"] = (lambda driver: click_element_by_xpath(driver, "//input[@value='0']", i))
    channel_selector[f"ch{i+1}_enabled"] = (lambda driver: click_element_by_xpath(driver, "//input[@value='1']", i))
    channel_selector[f"ch{i+1}_bias"] = (lambda driver: click_element_by_xpath(driver, "//input[@value='2']", i))


def set_up_tests(notebook_url, cell_index):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(notebook_url)

    # Wait for the notebook to load
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'notebook-container')))

    # Find the cell by index and execute it
    click_element_by_index(driver, 'inner_cell', cell_index)
    actions = ActionChains(driver)
    actions.key_down(Keys.CONTROL).send_keys(Keys.ENTER).key_up(Keys.CONTROL).perform()

    # Wait for the cell execution to complete
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'jupyter-button')))
    time.sleep(1)

    # Click on the channels section
    clickable_elements['channels_section'](driver)

    return driver

def tear_down_tests(driver):
    clickable_elements['stop'](driver)

    # Close the web driver
    driver.close()

def random_tests(driver, num_tests):

    selectable_options = list(clickable_elements.keys())
    options_to_remove = ["start", "stop", "active", "paused", "channels_section"]
    for option in options_to_remove:
        selectable_options.remove(option)

    for i in range(num_tests):
        print(f"################  TEST {i+1}/{num_tests}  ################")

        # Select four random channel setups:
        channel_setup = [f"ch{j+1}_{random.choice(['disabled', 'enabled', 'bias'])}" for j in range(NUM_CHANNELS)]
        for setup_opt in channel_setup:
            channel_selector[setup_opt](driver)
        print(f"Channel setup: {channel_setup}")

        # Generate a copy of the selectable options and shuffle it
        test_options = selectable_options.copy()
        random.shuffle(test_options)
        num_rem = random.randint(0, (len(test_options) - 1))
        test_options = test_options[:num_rem]
        for option in test_options:
            clickable_elements[option](driver)

        print(f"Selected Options: {test_options}")

        # Start the actual test:
        clickable_elements['start'](driver)
        time.sleep(1)
        clickable_elements['active'](driver)
        time.sleep(10)
        clickable_elements['paused'](driver)
        time.sleep(1)
        clickable_elements['stop'](driver)
        time.sleep(10)

        # Get the output
        output = driver.find_elements(By.CLASS_NAME, 'output')[0].text
        
        # Remove the lines that contain text from the widgets
        output = output.split("\n")[57:]

        # Check if the output is correct
        test_result = check_output(output)

        if not test_result:
            print("Test failed with output:")
            for line in output:
                print(line)
            return

    print("All tests passed!")
    

def run_notebook_cell(notebook_url, cell_index):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(notebook_url)

    # Wait for the notebook to load
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'notebook-container')))

    # Find the cell by index and execute it
    click_element_by_index(driver, 'inner_cell', cell_index)
    actions = ActionChains(driver)
    actions.key_down(Keys.CONTROL).send_keys(Keys.ENTER).key_up(Keys.CONTROL).perform()

    # Wait for the cell execution to complete
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'jupyter-button')))
    time.sleep(1)

    # Click on the channels section
    clickable_elements['channels_section'](driver)

    for clickable in clickable_elements:
        if clickable == "channels_section":
            continue
        clickable_elements[clickable](driver)
        time.sleep(1)

    clickable_elements['stop'](driver)

    # Retrieve the output
    output = driver.find_elements(By.CLASS_NAME, 'output')[cell_index].text

    # Close the web driver
    driver.close()

    return output


if __name__ == "__main__":
    cell_index = 0
    url = "http://192.168.4.1:8080/notebooks/portiloop-software/portiloop/notebooks/tests.ipynb"
    num_tests = 50

    print(f"Getting ready to launch {num_tests} tests...")
    driver = set_up_tests(url, cell_index)
    print(f"Testing should last an estimated {num_tests * 20} seconds to run...")
    random_tests(driver, num_tests)
    tear_down_tests(driver)




